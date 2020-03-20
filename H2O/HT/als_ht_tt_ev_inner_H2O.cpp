#include <ctime>
#include <xerus.h>
#include <queue>
#include <iterator>
#include <vector>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace xerus;
using xerus::misc::operator<<;

xerus::Tensor make_H_BE();
xerus::Tensor make_V_BE();
xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);

std::vector<size_t> cuthill_mckee(xerus::Tensor H);
xerus::Tensor make_Permutation(std::vector<size_t> _perm, size_t dim);

class InternalSolver {
	const size_t d;
	double lambda;
	double eps;
	size_t maxRank;
	const size_t p;

	std::vector<std::pair<Tensor,Tensor>> Stack;


	HTTensor& x;
	const TTOperator& A;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, HTTensor& _x, double _eps, size_t _maxRank)
		: d(_x.degree()), x(_x), A(_A),  maxIterations(1000), p(static_cast<size_t>(std::log(_x.degree()))), lambda(1.0), eps(_eps), maxRank(_maxRank),Stack(2*_x.degree()-2)
	{
	}

	double calc_residual_norm() {
		Index i,j,k;
		Tensor tmp;
		//tmp () = x(i&0)*A(i/2, j/2)*x(j&0) - lambda*x(i&0)*x(i&0);
		return frob_norm(A(i/2, j/2)*x(j&0) - lambda*x(i&0)) / frob_norm(x);
		//return tmp[0];
	}


	bool get_path_from_root(size_t _root, size_t _dest, std::vector<size_t>& _path ) const {
		if (_root > 2*d - 2) { return false;}
		_path.emplace_back(_root);
		if (_root == _dest) { return true;}
		if(get_path_from_root(_root*2+1,_dest,_path) || get_path_from_root(_root*2+2,_dest,_path)) {return true;}
		_path.pop_back();
		return false;

	}

	std::vector<size_t> get_path(size_t _start, size_t _end) const {
		std::vector<size_t> path_start;
		std::vector<size_t> path_end;
		std::vector<size_t> result;

		XERUS_REQUIRE(get_path_from_root(0, _start, path_start ), "start point is wrong");
		XERUS_REQUIRE(get_path_from_root(0, _end, path_end ), "end point is wrong");
		while(!path_start.empty()){
			size_t tmp = path_start.back();
			path_start.pop_back();
			auto tmp_found = std::find(path_end.begin(), path_end.end(), tmp);
			if (path_end.end() == tmp_found){ result.emplace_back(tmp);}
			else{
				result.insert(result.end(), tmp_found, path_end.end());
				break;
			}
		}
		std::reverse(result.begin(),result.end());
		return result;
	}


	void initizialize_stack(){
		Index i1, i2, i3,i4, i5, i6, j1 , j2, j3,j4, k1, k2,r1,r2;
		x.move_core(0);
		// first go from leaves to root
		// first update the leave edges
		size_t df = static_cast<size_t>(0.5+std::pow(2,std::ceil(std::log2(static_cast<double>(d)))));
		XERUS_LOG(info,df);
		for (size_t i = 2*d -3, j = d - 1; i >= d-2; --i){
			auto tto_comp_idx = i >= df - 2 ? i - df + 2 : j;
			j = i >= df - 2 ? j : j - 1;
			auto htt_comp_idx = i + 1;

			auto Ai = A.get_component(tto_comp_idx);
			auto xi = x.get_component(htt_comp_idx);
			Tensor tmp;
			//i1 out first x, i2 left output TTO, i3 right output TTO, i4 out second x
			tmp(i1,i2,i3,i4) = xi(i1,j2) * xi(i4,j3) * Ai(i2,j2,j3,i3);
			Stack[i].first = tmp;
		}
		// then the internal components till the edge number 2 because we start traversing from root
		for (int i = d - 3; i >= 0; --i){
			auto htt_comp_idx = i + 1;
			auto xi = x.get_component(htt_comp_idx);
  		auto lchild = Stack[2*htt_comp_idx].first;
			auto rchild = Stack[2*htt_comp_idx + 1].first;
			Tensor tmp;
			//i1 out first x, i2 left output TTO, i3 right output TTO, i4 out second x
			tmp(i1,i2,i3,i4) = xi(i1,j1,j2) * xi(i4,j3,j4) * lchild(j1,i2,r1,j3) * rchild(j2,r1,i3,j4);
			Stack[i].first = tmp;
		}
	}
	void update_stack_up(size_t _edge){
		Index i1, i2, i3,i4, i5, i6, j1 , j2, j3,j4, k1, k2,r1,r2;
		auto htt_comp_idx = _edge / 2;
		Tensor tmp1,tmp2;
		auto xi = x.get_component(htt_comp_idx);
		if (_edge < 2){
			auto dummy = Tensor::dirac({1},0);
			if (_edge == 0){
				auto rchild = Stack[1].first;
				tmp1(i1,i2,i3,i4) = dummy(r1) * dummy(r2) * xi(r1,i1,j2) * xi(r2,i4,j4)  * rchild(j2,i2,i3,j4);
				Stack[0].second = tmp1;

			} else {
				auto lchild = Stack[0].first;
				tmp2(i1,i2,i3,i4) = dummy(r1) * dummy(r2) * xi(r1,j2,i1) * xi(r2,j4,i4)  * lchild(j2,i2,i3,j4);
				Stack[1].second = tmp2;
			}
		} else {
			auto parent = Stack[htt_comp_idx - 1].second;
			if (_edge % 2 == 0){
				auto rchild = Stack[2*htt_comp_idx + 1].first;
				if (std::abs(std::log2(_edge+2) - std::floor(std::log2(_edge+2))) < 10e-12){
					tmp1(i1,i2,i3,i4) = parent(r1,j3,i3,r2) * xi(r1,i1,j2) * xi(r2,i4,j4)  * rchild(j2,i2,j3,j4);
				} else if (std::abs(std::log2(_edge+4) - std::floor(std::log2(_edge+4))) < 10e-12){
					tmp1(i1,i2,i3,i4,i5,i6) = parent(r1,i2,i3,r2) * xi(r1,i1,j2) * xi(r2,i6,j4)  * rchild(j2,i4,i5,j4);
				} else {
					tmp1(i1,i2,i3,i4,i5,i6) = parent(r1,i2,i3,k1,i5,r2) * xi(r1,i1,j2) * xi(r2,i6,j4)  * rchild(j2,i4,k1,j4);
				}
				Stack[_edge].second = tmp1;
			} else {
				auto lchild = Stack[2*htt_comp_idx].first;
				if (std::abs(std::log2(_edge+3) - std::floor(std::log2(_edge+3))) < 10e-12){
					tmp2(i1,i2,i3,i4) = parent(r1,i2,j3,r2) * xi(r1,j2,i1) * xi(r2,j4,i4)  * lchild(j2,j3,i3,j4);
				} else if (std::abs(std::log2(_edge+1) - std::floor(std::log2(_edge+1))) < 10e-12){
					tmp2(i1,i2,i3,i4,i5,i6) = parent(r1,i4,i5,r2) * xi(r1,j2,i1) * xi(r2,j4,i6)  * lchild(j2,i2,i3,j4);
				} else {
					tmp2(i1,i2,i3,i4,i5,i6) = parent(r1,i2,k1,i4,i5,r2) * xi(r1,j2,i1) * xi(r2,j4,i6)  * lchild(j2,k1,i3,j4);
				}
				Stack[_edge].second = tmp2;
			}
		}
	}

	void update_stack_down(size_t _edge){
		Index i1, i2, i3,i4, i5, i6, j1 , j2, j3,j4, k1, k2,r1,r2;
		auto htt_comp_idx = _edge + 1;
		auto xi = x.get_component(htt_comp_idx);
		auto lchild = Stack[2*htt_comp_idx].first;
		auto rchild = Stack[2*htt_comp_idx + 1].first;
		Tensor tmp;
		//i1 out first x, i2 left output TTO, i3 right output TTO, i4 out second x
		tmp(i1,i2,i3,i4) = xi(i1,j1,j2) * xi(i4,j3,j4) * lchild(j1,i2,r1,j3) * rchild(j2,r1,i3,j4);
			Stack[_edge].first = tmp;
	}

	double solve() {
		if (d < 4){
			XERUS_LOG(error, "d is to small");
			return 0;
		}

		//auto trav = calc_traversal();
		clock_t start;
		value_t es_move_inner = 0.0,es_move_root = 0.0,es_contract_inner = 0.0,es_contract_root = 0.0,es_solve_inner = 0.0;
		value_t es_solve_root = 0.0,es_solve2_inner = 0.0,es_solve2_root = 0.0,es_svd_inner = 0.0,es_svd_root = 0.0;
		Index i1, i2, i3,i4, j1 , j2, j3,j4, k1, k2,k3,k4,r1,r2;

		start = std::clock();
		initizialize_stack();

		es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

		std::vector<double> residuals(10, 1000.0);
		std::vector<double> residuals2(10, 1000.0);
		auto last_cP = 0;

		for (size_t itr = 0; itr < maxIterations; ++itr) {

			// Calculate residual and check end condition
			//residuals.push_back(calc_residual_norm());
			residuals2.push_back(lambda);
			if (itr > 3 && residuals2[residuals2.size()-3] / residuals2.back() > 0.99999) {
				//XERUS_LOG(simpleALS, "Done! Residual is now " << std::scientific << calc_residual_norm());
				XERUS_LOG(info, "Inner Move:     " << es_move_inner << " \tsec");
				XERUS_LOG(info, "Inner Contract: " << es_contract_inner << " \tsec");
				XERUS_LOG(info, "Inner Solve:    " << es_solve_inner << " \tsec");
				//XERUS_LOG(info, "Inner Solve2:   " << es_solve2_inner << " \tsec");
				XERUS_LOG(info, "Inner SVD:      " << es_svd_inner << " \tsec");
				XERUS_LOG(info, "Root  Move:     " << es_move_root << " \tsec");
				XERUS_LOG(info, "Root  Contract: " << es_contract_root << " \tsec");
				XERUS_LOG(info, "Root  Solve:    " << es_solve_root << " \tsec");
				//XERUS_LOG(info, "Root  Solve2:   " << es_solve2_root << " \tsec");
				XERUS_LOG(info, "Root  SVD:      " << es_svd_root << " \tsec");
				auto total = es_move_root+es_move_inner+es_contract_root+es_contract_inner+es_solve_root+es_solve_inner + es_svd_root+es_svd_inner;
				XERUS_LOG(info, "Move:           " << es_move_root+es_move_inner << " \tsec " << 100*(es_move_root+es_move_inner)/total << " \t%");
				XERUS_LOG(info, "Contract:       " << es_contract_root+es_contract_inner << " \tsec "<< 100*(es_contract_root+es_contract_inner)/total << " \t%");
				XERUS_LOG(info, "Solve:          " << es_solve_root+es_solve_inner << " \tsec "<< 100*(es_solve_root+es_solve_inner)/total << " \t%");
				XERUS_LOG(info, "SVD:            " << es_svd_root+es_svd_inner << " \tsec "<< 100*(es_svd_root+es_svd_inner)/total << " \t%");
				return lambda; // We are done!
			}
			XERUS_LOG(simpleALS, "It:  " << itr  << " EV " << lambda -28.1930439210 << " Lerr =  "  << std::abs(lambda -28.1930439210	+38.979392539208));
			XERUS_LOG(simpleALS, "Sol Ranks:  " << x.ranks());
			//root component
			start = std::clock();
			std::vector<size_t> 	path = get_path(last_cP,0);
			while (path.size() > 1){
				size_t start = path.back();
				path.pop_back();
				size_t end = path.back();
				if (start > end){
					x.move_core(end,true);
					update_stack_down(start - 1);
				}
			}
			es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

			TensorNetwork op,op2;

			start = std::clock();
			auto dummy = Tensor::dirac({1},0);

			op (i1,i2,j1,j2) =  dummy(r1) * dummy(r2) * \
																			Stack[0].first(i1,r1,k1,j1) * Stack[1].first(i2,k1,r2,j2);
			es_contract_root += double(std::clock() - start) / CLOCKS_PER_SEC;
			Tensor sol;
			Tensor &xi = x.component(0);

			sol(i1,i2) = dummy(j1)*xi(j1,i1,i2);
			start = std::clock();
	  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
			xerus::get_smallest_eigenvalue_iterative(sol,op,ev.get(), 1, 100000, 10e-6);
	  	lambda = ev[0];
			es_solve_root += double(std::clock() - start) / CLOCKS_PER_SEC;

			start = std::clock();
			es_svd_root += double(std::clock() - start) / CLOCKS_PER_SEC;

			sol.reinterpret_dimensions({1,sol.dimensions[0],sol.dimensions[1]});
			x.set_component(0,sol);

			x.assume_core_position(0);
			x.move_core(1);
			last_cP = 1;
			start = std::clock();
			update_stack_down(1);
			update_stack_up(0);
			es_contract_root += double(std::clock() - start) / CLOCKS_PER_SEC;
			XERUS_LOG(simpleALS, "It:  " << itr << " CP " << 0  << " EV " << lambda +8.80146457125193);

			for (size_t corePosition : {1,3,7,15,16,8,17,18,4,9,19,20,10,21,22,2,5,11,23,24,12,6,13,14}) {


				//move core
				start = std::clock();
				std::vector<size_t> 	path = get_path(last_cP,corePosition);
				while (path.size() > 1){
					size_t start = path.back();
					path.pop_back();
					size_t end = path.back();
					x.move_core(end,true);
					if (start > end){
						update_stack_down(start - 1);
					}
					if (end > start){
						x.move_core(end,true);
						update_stack_up(end - 1);
					}
				}
				es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				start = std::clock();
				x.move_core(corePosition, true);
				es_move_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				//build operator
				start = std::clock();
				auto dummy = Tensor::dirac({1},0);
				auto lc = 2 * corePosition;
				auto rc = 2 * corePosition + 1;
				auto pp = corePosition-1;
				if (Stack[pp].second.degree() == 6){
					op (i1,i2,i3,j1,j2,j3) =  dummy(r1) * dummy(r2) * \
																		Stack[pp].second(i1,r1,k1,k3,r2,j1) * Stack[lc].first(i2,k1,k2,j2) * \
																		Stack[rc].first (i3,k2,k3,j3);

				} else
						if(pp % 2 == 1)
							op (i1,i2,i3,j1,j2,j3) =  dummy(r1) * dummy(r2) * \
																				Stack[pp].second(i1,r1,k1,j1) * Stack[lc].first(i2,k1,k2,j2) * \
																				Stack[rc].first (i3,k2,r2,j3);
						else
							op (i1,i2,i3,j1,j2,j3) =  dummy(r1) * dummy(r2) * \
																				Stack[pp].second(i1,k2,r2,j1) * Stack[lc].first(i2,r1,k1,j2) * \
																				Stack[rc].first (i3,k1,k2,j3);
				//op = op2;
				es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;
				//solve local problem
				start = std::clock();
		  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
				xerus::get_smallest_eigenvalue_iterative(x.component(corePosition),op,ev.get(), 1, 100000, 1e-10);
		  	lambda = ev[0];
				es_solve_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				last_cP = corePosition;
				XERUS_LOG(simpleALS, "It:  " << itr << " CP " << corePosition  << " EV " << lambda + 8.80146457125193 );

			}
		}
		return lambda;
	}

};

double simpleALS(const TTOperator& _A, HTTensor& _x, double _eps, size_t _maxRank)  {
	InternalSolver solver(_A, _x, _eps, _maxRank);
	return solver.solve();
}


int main() {
  xerus::Index i1,i2,i3,i4, ii,jj,kk,ll;


	Index i,j,k;
	auto dim = 50;
	double eps = 10e-7;
	std::vector<size_t> start_rank = std::vector<size_t>(2*dim-2,8);
	size_t max_rank = 12;


	auto op = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
	std::string name = "../hamiltonian50_full.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();

	XERUS_LOG(info, "The ranks of op are " << op.ranks() );

	xerus::HTTensor phi = xerus::HTTensor::random(std::vector<size_t>(dim,2),start_rank);
	double lambda = simpleALS(op, phi, eps, max_rank);//
	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	//XERUS_LOG(info, "Residual: " << frob_norm(op(i/2, j/2) * phi(j&0) - lambda * phi(i&0))/frob_norm(phi));
	XERUS_LOG(info, "Lambda =  " << std::setprecision(16) << lambda + 8.80146457125193	);



}



xerus::Tensor make_H_BE(){
	auto H = xerus::Tensor({16,16});
	auto H_tmp = xerus::Tensor({8,8});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.be_48");
	size_t count = 0;
	while ( std::getline (input,line) )
	{
		count++;
		if (count > 4){
			std::vector<std::string> l;
			boost::algorithm::split_regex( l, line, boost::regex( "   " ) ) ;
			if (std::stoi(l[1]) != 0 && std::stoi(l[3]) == 0){
				H_tmp[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1}] = stod(l[0]);
	    }
		}
	}
	input.close();
	for (size_t i = 0; i < 8; i++){
		for (size_t j = 0; j < 8; j++){
			auto val = H_tmp[{i,j}];
			H[{2*i,2*j}] = val;
			H[{2*j,2*i}] = val;
			H[{2*i+1,2*j+1}] = val;
			H[{2*j+1,2*i+1}] = val;
		}
	}
	return H;
}


xerus::Tensor make_V_BE(){
	auto V = xerus::Tensor({16,16,16,16});
	auto V_tmp = xerus::Tensor({8,8,8,8});
	auto V_tmp2 = xerus::Tensor({8,8,8,8});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.be_48");
	size_t count = 0;
	while ( std::getline (input,line) )
	{
		count++;
		if (count > 4){
			std::vector<std::string> l;
			boost::algorithm::split_regex( l, line, boost::regex( "   " ) ) ;
			if (std::stoi(l[1]) != 0 && std::stoi(l[3]) != 0){
				V_tmp[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1,static_cast<size_t>(std::stoi(l[3]))-1,static_cast<size_t>(std::stoi(l[4]))-1}] = stod(l[0]);
			}
		}
	}
	input.close();
	for (size_t i = 0; i < 8; i++){
			for (size_t j = 0; j <= i; j++){
				for (size_t k = 0; k<= i; k++){
					for (size_t l = 0; l <= (i==k ? j : k); l++){
						auto value = V_tmp[{i,j,k,l}];
						V_tmp2[{i,k,j,l}] = value;
						V_tmp2[{j,k,i,l}] = value;
						V_tmp2[{i,l,j,k}] = value;
						V_tmp2[{j,l,i,k}] = value;
						V_tmp2[{k,i,l,j}] = value;
						V_tmp2[{l,i,k,j}] = value;
						V_tmp2[{k,j,l,i}] = value;
						V_tmp2[{l,j,k,i}] = value;
					}
				}
			}
	}
	for (size_t i = 0; i < 8; i++){
				for (size_t j = 0; j < 8; j++){
					for (size_t k = 0; k < 8; k++){
						for (size_t l = 0; l < 8; l++){
							auto value = V_tmp2[{i,j,k,l}];
							V[{2*i,2*j,2*k,2*l}] = value;
							V[{2*i+1,2*j,2*k+1,2*l}] = value;
							V[{2*i,2*j+1,2*k,2*l+1}] = value;
							V[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
						}
					}
				}
	}

	return V;
}
xerus::Tensor make_Permutation(std::vector<size_t> _perm, size_t dim){
	auto perm = xerus::Tensor({dim,dim});
	for (size_t i = 0; i < dim; ++i)
		perm[{i,_perm[i]}] = 1;
	return perm;
}

std::vector<size_t> cuthill_mckee(xerus::Tensor H){
	std::queue<size_t> Q;
	std::vector<size_t> R;
	auto d = H.dimensions[0];
	std::vector<bool> visited(d,false);
	auto eps = 10e-10;
	// get degrees
	std::vector<std::vector<size_t>> deg(d);
	for (size_t i = 0; i < d;++i){
		for (size_t j = 0; j < d; ++j)
			if(std::abs(H[{i,j}]) > eps)
				deg[i].emplace_back(j);
	}
	XERUS_LOG(info, "deg = " << deg);
	//fill queue
	while(R.size() < d){
		size_t ind = -1;
		size_t min = d;
		for(size_t i = 0; i < d; ++i){
			if(deg[i].size() < min && std::find(R.begin(), R.end(), i) == R.end()){
				min = deg[i].size();
				ind = i;
			}
		}
		Q.push(ind);
		visited[ind] = true;
		while(!Q.empty()){
			ind = Q.front();
			R.emplace_back(ind);
			for(size_t i = 0; i < min; ++i){
				if(!visited[deg[ind][i]]){
					Q.push(deg[ind][i]);
					visited[deg[ind][i]] = true;
				}
			}
			Q.pop();
		}
	}
	std::reverse(R.begin(),R.end());
	return R;
}

//Creation of Operators
xerus::TTOperator return_annil(size_t i, size_t d){ // TODO write tests for this


	xerus::Index i1,i2,jj, kk, ll;
	auto a_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto annhil = xerus::Tensor({2,2});
	annhil[{0,1}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? s : (m == i ? annhil : id );
		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];
		a_op.set_component(m, res);
	}
	return a_op;
}

xerus::TTOperator return_create(size_t i, size_t d){ // TODO write tests for this
	xerus::Index i1,i2,jj, kk, ll;
	auto c_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

	auto id = xerus::Tensor({2,2});
	id[{0,0}] = 1.0;
	id[{1,1}] = 1.0;
	auto s = xerus::Tensor({2,2});
	s[{0,0}] = 1.0;
	s[{1,1}] = -1.0;
	auto create = xerus::Tensor({2,2});
	create[{1,0}] = 1.0;
	for (size_t m = 0; m < d; ++m){
		auto tmp = m < i ? s : (m == i ? create : id );
		auto res = xerus::Tensor({1,2,2,1});
		res[{0,0,0,0}] = tmp[{0,0}];
		res[{0,1,1,0}] = tmp[{1,1}];
		res[{0,1,0,0}] = tmp[{1,0}];
		res[{0,0,1,0}] = tmp[{0,1}];
		c_op.set_component(m, res);
	}
	return c_op;
}

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d){ // TODO write tests for this
	auto cr = return_create(i,d);
	auto an = return_annil(j,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk;
	res(ii/2,jj/2) = cr(ii/2,kk/2) * an(kk/2, jj/2);
	return res;
}

xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d){ //todo test
	auto cr1 = return_create(i,d);
	auto cr2 = return_create(j,d);
	auto an1 = return_annil(k,d);
	auto an2 = return_annil(l,d);
	xerus::TTOperator res;
	xerus::Index ii,jj,kk,ll,mm;
	res(ii/2,mm/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) * an2(ll/2,mm/2);
	return res;
}

xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim){
	auto res = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	if (a!=b && c!=d)
		res += return_two_e_ac(a,b,d,c,dim);
		res += return_two_e_ac(a+1,b+1,d+1,c+1,dim);
	if (c != d)
		res += return_two_e_ac(a+1,b,d,c+1,dim);
	res += return_two_e_ac(a,b+1,d+1,c,dim);


	return res;
}
