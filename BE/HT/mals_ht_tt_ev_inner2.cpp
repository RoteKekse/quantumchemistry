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

	std::vector<Tensor> leftAStack;
	std::vector<TensorNetwork> rightAStack;

	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;

	HTTensor& x;
	const TTOperator& A;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, HTTensor& _x, double _eps, size_t _maxRank)
		: d(_x.degree()), x(_x), A(_A),  maxIterations(1000), p(static_cast<size_t>(std::log(_x.degree()))), lambda(1.0), eps(_eps), maxRank(_maxRank)
	{
	}

	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - lambda*x(i&0)) / frob_norm(x);

	}

	std::vector<size_t> calc_traversal(){
		std::vector<size_t> trav;
		trav.emplace_back(2*d-2);
		auto elm = trav.back();
		while (trav.size() < 2*d-1){
				if (std::find(trav.begin(), trav.end(), 2*elm + 2) == trav.end() && 2*elm + 2 < 2*d - 1){
									elm = 2*elm + 2;
				}
				else if (std::find(trav.begin(), trav.end(), 2*elm + 1) == trav.end() && 2*elm + 1 < 2*d - 1){
									elm = 2*elm + 1;
				}
				else if (std::find(trav.begin(), trav.end(), elm) != trav.end()){
					elm = (elm-1) / 2;
					if(std::find(trav.begin(), trav.end(), elm) == trav.end()){
						trav.emplace_back(elm);
					}
				}
				else{
					trav.emplace_back(elm);
				}
		}
		return trav;
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
		Index i1, i2, i3,i4, j1 , j2, j3,j4, k1, k2,r1,r2;
		std::vector<double> residuals(10, 1000.0);
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			residuals.push_back(calc_residual_norm());
			if (itr > 10 && residuals.back()/residuals[residuals.size()-10] > 0.99999) {
				XERUS_LOG(simpleALS, "Done! Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
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
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back() << " Eigenvalue " << lambda);
			for (size_t corePosition : {6,5,4,3}) {

				//move core
				start = std::clock();
				x.move_core(corePosition, true);
				es_move_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				TensorNetwork optn;
				Tensor op;

				//contract network
				start = std::clock();
				optn() =   x(i1^d)*x(j1^d) *  A(i1^d,j1^d);
				optn.nodes[corePosition].erase();
				optn.nodes[(corePosition-1)/2].erase();
				optn.nodes[corePosition+2*d].erase();
				optn.nodes[(corePosition-1)/2 + 2*d].erase();
				optn.sanitize();
				op (i1,i2,i3,i4,j1,j2,j3,j4) = optn(i1,i2,i3,i4,j1,j2,j3,j4);
				es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				Tensor sol, xup, xdown;
				Tensor U, S, Vt;

				//solve local problem
				start = std::clock();
		  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
				xerus::get_smallest_eigenvalue_iterative(sol,op,ev.get(), 1, 100000, 1e-10);
		  	lambda = ev[0];
				es_solve_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

//				start = std::clock();
//				lambda = xerus::get_smallest_eigenvalue(sol, op);
//				es_solve2_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				//SVD
				start = std::clock();
				if (corePosition % 2 == 1){
					(U(i1,r1,i2), S(r1,r2), Vt(r2,i3,i4)) = SVD(sol(i1,i2,i3,i4),maxRank,eps);
					xdown(i1,i2,i3) = U(i1,r1,i3) *  S(r1,i2);
				}
				else {
					(U(i1,i2,r1), S(r1,r2), Vt(r2,i3,i4)) = SVD(sol(i1,i2,i3,i4),maxRank,eps);
					xdown(i1,i2,i3) = U(i1,i2,r1) *  S(r1,i3);
				}
				es_svd_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				XERUS_LOG(info, "S_" << corePosition << " =  ");
				for ( size_t i =0; i < S.dimensions[0];++i)
									XERUS_LOG(info,std::setprecision(16) << S[{i,i}]);
				std::vector<std::string> S_inner;
				for ( size_t i =0; i < S.dimensions[0];++i)
					S_inner.emplace_back(std::to_string(S[{i,i}]));
				std::ofstream output_file("./S_" + std::to_string(corePosition)+".csv");
				std::ostream_iterator<std::string> output_iterator(output_file, "\n");
				std::copy(S_inner.begin(), S_inner.end(), output_iterator);

				x.set_component(corePosition,Vt);
				x.set_component((corePosition-1)/2,xdown);
				x.assume_core_position((corePosition-1)/2);

			}
			// root
			start = std::clock();
			x.move_core(0, true);
			es_move_root += double(std::clock() - start) / CLOCKS_PER_SEC;


			TensorNetwork optn;
			Tensor op;

			start = std::clock();
			optn() =   x(i1^d)*x(j1^d) *  A(i1^d,j1^d);
			optn.nodes[0].erase();
			optn.nodes[1].erase();
			optn.nodes[2].erase();
			optn.nodes[2*d-1].erase();
			optn.nodes[2*d].erase();
			optn.nodes[2*d+1].erase();
			optn.nodes[2*d+2].erase();
			optn.nodes[4*d-1].erase();
			optn.sanitize();
			op (i1,i2,i3,i4,j1,j2,j3,j4) = optn(i1,i2,i3,i4,j1,j2,j3,j4);
			es_contract_root += double(std::clock() - start) / CLOCKS_PER_SEC;

			Tensor sol;

			start = std::clock();
	  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
			xerus::get_smallest_eigenvalue_iterative(sol,op,ev.get(), 1, 100000, 1e-10);
	  	lambda = ev[0];
			es_solve_root += double(std::clock() - start) / CLOCKS_PER_SEC;

//			start = std::clock();
//			lambda = xerus::get_smallest_eigenvalue(sol, op);
//			es_solve2_root += double(std::clock() - start) / CLOCKS_PER_SEC;

			Tensor U, S, Vt;

			start = std::clock();
			(U(r1,i1,i2), S(r1,r2), Vt(r2,i3,i4)) = SVD(sol(i1,i2,i3,i4),maxRank,eps);
			es_svd_root += double(std::clock() - start) / CLOCKS_PER_SEC;
			XERUS_LOG(info, "S_root = ");
			for ( size_t i =0; i < S.dimensions[0];++i)
				XERUS_LOG(info,std::setprecision(16) << S[{i,i}]);
			std::vector<std::string> S_root;
			for ( size_t i =0; i < S.dimensions[0];++i)
				S_root.emplace_back(std::to_string(S[{i,i}]));
	    std::ofstream output_file("./S_root.csv");
	    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
	    std::copy(S_root.begin(), S_root.end(), output_iterator);

			x.set_component(1,U);
			x.set_component(2,Vt);
			S.reinterpret_dimensions({1,S.dimensions[0],S.dimensions[1]});
			x.set_component(0,S);
			x.assume_core_position(0);

			x.move_core(d-2, true);

		}
		return lambda;
	}

};

double simpleMALS(const TTOperator& _A, HTTensor& _x, double _eps, size_t _maxRank)  {
	InternalSolver solver(_A, _x, _eps, _maxRank);
	return solver.solve();
}


int main() {
  xerus::Index i1,i2,i3,i4, ii,jj,kk,ll;


	Index i,j,k;
	auto dim = 8;
	double eps = 0.0000000000000000001;
	size_t start_rank = 2;
	size_t max_rank = 250;

	xerus::Tensor H = make_H_BE();
	XERUS_LOG(info,"\n"<< H);
  xerus::Tensor V = make_V_BE();

  std::string name2 = "V.tensor";
	std::ofstream write(name2.c_str() );
	xerus::misc::stream_writer(write,V,xerus::misc::FileFormat::BINARY);
	write.close();

	std::vector<size_t> perm = cuthill_mckee(H);
	XERUS_LOG(simpleMALS,"Permutation = " << perm);
	//xerus::Tensor perm_T = make_Permutation(perm, dim);
//	xerus::Tensor perm_T = make_Permutation({7,5,3,1,6,4,2,0}, dim);
//	H(i1,i2) = perm_T(i1,jj) * H(jj,kk) *perm_T(i2,kk);
//  V(i1,i2,i3,i4) = perm_T(i1,ii) * perm_T(i2,jj) * perm_T(i3,kk) * perm_T(i4,ll) * V(ii,jj,kk,ll);



	auto opH = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	for (size_t i = 0; i < dim; i++){
			for (size_t j = 0; j < dim; j++){
					value_t val = H[{i , j}];
					opH += val * return_one_e_ac(i,j,dim);
			}
	}
	auto opV = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	for (size_t i = 0; i < dim; i++){
		for (size_t j = 0; j < dim; j++){
			for (size_t k = 0; k < dim; k++){
				for (size_t l = 0; l < dim; l++){
					value_t val = V[{i,j,k,l}];
					if (std::abs(val) < eps )//|| a == b || c: == d) // TODO check i == j || k == l
						continue;
					opV += 0.5*val * return_two_e_ac(i,j,l,k,dim);
				}
			}
		}
	}

  xerus::TTOperator op = opH + opV;
	xerus::HTTensor phi = xerus::HTTensor::random(std::vector<size_t>(dim,2),std::vector<size_t>(2*dim - 2,start_rank));
	double lambda = simpleMALS(op, phi, eps, max_rank);//
	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	XERUS_LOG(info, "Residual: " << frob_norm(op(i/2, j/2) * phi(j&0) - lambda * phi(i&0))/frob_norm(phi));
	XERUS_LOG(info, "Lambda =  " << std::setprecision(16) << lambda);
	XERUS_LOG(info, "Lambda Error =  " << std::setprecision(16) << std::abs(lambda	+14.39207049621482));

//	xerus::HTTensor x_HT = xerus::HTTensor::random(std::vector<size_t>(16,2),std::vector<size_t>(2*16 - 2,256));
//	xerus::TTTensor x_TT = xerus::TTTensor::random(std::vector<size_t>(16,2),std::vector<size_t>(16 - 1,256));
//	XERUS_LOG(info, "x_HT =  " << x_HT.dimensions << " " << x_HT.ranks());
//	XERUS_LOG(info, "x_TT =  " << x_TT.dimensions << " " << x_TT.ranks());

}


xerus::Tensor make_H_BE(){
	auto H = xerus::Tensor({8,8});
	auto H_tmp = xerus::Tensor({4,4});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.be_44");
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
	for (size_t i = 0; i < 4; i++){
		for (size_t j = 0; j < 4; j++){
			auto val = H_tmp[{i,j}];
			H[{2*i,2*j}] = val;
			H[{2*j,2*i}] = val;
			H[{2*i+1,2*j+1}] = val;
			H[{2*j+1,2*i+1}] = val;
		}
	}
	return H;
}
//xerus::Tensor make_H_BE(){
//	auto H = xerus::Tensor({4,4});
//	auto H2 = xerus::Tensor({8,8});
//  H[{0,0}] =      -7.854546469;
//  H[{1,0}] =   		0.2190509890;
//  H[{0,1}] =   		0.2190509890;
//  H[{1,1}] =      -1.594069613;
//	H[{2,0}] =    -0.5038662979E-01;
//	H[{0,2}] =    -0.5038662979E-01;
//	H[{2,1}] =     0.1590245167;
//	H[{1,2}] =     0.1590245167;
//	H[{2,2}] =    -0.4996472436;
//	H[{3,0}] =     0.2158425412;
//	H[{0,3}] =     0.2158425412;
//	H[{3,1}] =    -0.4395272886;
//	H[{1,3}] =    -0.4395272886;
//	H[{3,2}] =     0.2151398215;
//	H[{2,3}] =     0.2151398215;
//	H[{3,3}] =    -0.9315770207;
//	for (size_t i = 0; i < 4; i++){
//		for (size_t j = 0; j < 4; j++){
//			auto value = H[{i,j}];
//				H2[{2*i,2*j,}] = value;
//				H2[{2*i+1,2*j+1}] = value;
//		}
//	}
//	return H2;
//}

xerus::Tensor make_V_BE(){
	auto V = xerus::Tensor({8,8,8,8});
	auto V_tmp = xerus::Tensor({4,4,4,4});
	auto V_tmp2 = xerus::Tensor({4,4,4,4});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.be_44");
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
	for (size_t i = 0; i < 4; i++){
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
	for (size_t i = 0; i < 4; i++){
				for (size_t j = 0; j < 4; j++){
					for (size_t k = 0; k < 4; k++){
						for (size_t l = 0; l < 4; l++){
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

//xerus::Tensor make_V_BE(){
//	auto V = xerus::Tensor({4,4,4,4});
//	auto V_end = xerus::Tensor({4,4,4,4});
//	auto V_end2 = xerus::Tensor({8,8,8,8});
//	V[{0,0,0,0}] =  2.267945766    ;
//	V[{1,0,0,0}] = -0.2111818790    ;
//	V[{1,0,1,0}] = 0.3138258308E-01;
//	V[{1,1,0,0}] = 0.4909715652;
//	V[{1,1,1,0}] = -0.7869113237E-02;
//	V[{1,1,1,1}] = 0.3397511542;
//	V[{2,0,0,0}] = 0.4895090312E-01;
//	V[{2,0,1,0}] = -0.7343246602E-02;
//	V[{2,0,1,1}] = 0.1646118471E-02;
//	V[{2,0,2,0}] = 0.1719084918E-02;
//	V[{2,1,0,0}] = -0.6718385965E-01;
//	V[{2,1,1,0}] = 0.1857595751E-02;
//	V[{2,1,1,1}] = -0.3313262761E-01;
//	V[{2,1,2,0}] = -0.4265413154E-03;
//	V[{2,1,2,1}] = 0.6222837139E-02;
//	V[{2,2,0,0}] = 0.1416413337    ;
//	V[{2,2,1,0}] = -0.4337337159E-03;
//	V[{2,2,1,1}] = 0.1337132305    ;
//	V[{2,2,2,0}] = 0.2441914512E-04 ;
//	V[{2,2,2,1}] = -0.1834373106E-02 ;
//	V[{2,2,2,2}] = 0.1100227758    ;
//	V[{3,0,0,0}] = -0.2072195262;
//	V[{3,0,1,0}] = 0.3091786988E-01;
//	V[{3,0,1,1}] = -0.8269120939E-02;
//	V[{3,0,2,0}] = -0.7234672550E-02;
//	V[{3,0,2,1}] = 0.1882381970E-02;
//	V[{3,0,2,2}] = -0.4995136405E-03;
//	V[{3,0,3,0}] = 0.3047932126E-01;
//	V[{3,1,0,0}] = 0.1997162253 ;
//	V[{3,1,1,0}] = -0.7920083590E-02;
//	V[{3,1,1,1}] = 0.7115559280E-01;
//	V[{3,1,2,0}] = 0.1832178095E-02;
//	V[{3,1,2,1}] = -0.1504715160E-01;
//	V[{3,1,2,2}] = 0.3567543346E-02 ;
//	V[{3,1,3,0}] = -0.7669371553E-02;
//	V[{3,1,3,1}] = 0.4423261061E-01;
//	V[{3,2,0,0}] = -0.7453142107E-01;
//	V[{3,2,1,0}] = 0.1849152148E-02;
//	V[{3,2,1,1}] = -0.4417940602E-01;
//	V[{3,2,2,0}] = -0.3834295785E-03 ;
//	V[{3,2,2,1}] = 0.4937902093E-02 ;
//	V[{3,2,2,2}] = -0.9589124471E-02;
//	V[{3,2,3,0}] = 0.1861810917E-02;
//	V[{3,2,3,1}] = -0.1106087118E-01;
//	V[{3,2,3,2}] = 0.8838257752E-02  ;
//	V[{3,3,0,0}] = 0.3890399635    ;
//	V[{3,3,1,0}] = -0.7845863700E-02;
//	V[{3,3,1,1}] = 0.2610468864;
//	V[{3,3,2,0}] = 0.1664841315E-02;
//	V[{3,3,2,1}] = -0.2047995105E-01;
//	V[{3,3,2,2}] = 0.1250626594;
//	V[{3,3,3,0}] = -0.7894000039E-02;
//	V[{3,3,3,1}] = 0.4583223164E-01 ;
//	V[{3,3,3,2}] = -0.3258593753E-01;
//	V[{3,3,3,3}] = 0.2136542350 ;
//	for (size_t i = 0; i < 4; i++){
//			for (size_t j = 0; j <= i; j++){
//				for (size_t k = 0; k<= i; k++){
//					for (size_t l = 0; l <= (i==k ? j : k); l++){
//						auto value = V[{i,j,k,l}];
//						V_end[{i,k,j,l}] = value;
//						V_end[{j,k,i,l}] = value;
//						V_end[{i,l,j,k}] = value;
//						V_end[{j,l,i,k}] = value;
//						V_end[{k,i,l,j}] = value;
//						V_end[{l,i,k,j}] = value;
//						V_end[{k,j,l,i}] = value;
//						V_end[{l,j,k,i}] = value;
//					}
//				}
//			}
//	}
//	for (size_t i = 0; i < 4; i++){
//				for (size_t j = 0; j < 4; j++){
//					for (size_t k = 0; k < 4; k++){
//						for (size_t l = 0; l < 4; l++){
//							auto value = V_end[{i,j,k,l}];
//							V_end2[{2*i,2*j,2*k,2*l}] = value;
//							V_end2[{2*i+1,2*j,2*k+1,2*l}] = value;
//							V_end2[{2*i,2*j+1,2*k,2*l+1}] = value;
//							V_end2[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
//						}
//					}
//				}
//	}
//
//	return V_end2;
//}
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
