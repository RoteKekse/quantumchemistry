#include <xerus.h>
#include <Eigen/Dense>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>

#define build_operator 0
#if build_operator
#include <mpi.h>
#endif

using namespace xerus;
using xerus::misc::operator<<;

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage

template<typename M>
M load_csv (const std::string & path);

xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);

xerus::TTOperator return_op_H(xerus::Tensor H);
xerus::Tensor make_H(size_t nob);
xerus::Tensor make_H_BE();
xerus::Tensor make_V_BE();

#if build_operator
xerus::Tensor make_V(size_t nob);
xerus::TTOperator return_op_V(xerus::Tensor V);
#endif
xerus::Tensor make_Permutation(std::vector<size_t> _perm, size_t dim);

std::vector<size_t> cuthill_mckee(xerus::Tensor H);

class InternalSolver;
class InternalSolver2;
double simpleALS(const TTOperator& _A, TTTensor& _x);
double simpleMALS(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank);


/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");
  xerus::Index i1,i2,i3,i4, ii,jj,kk,ll;


	size_t dim = 8; // 16 electron, 8 electron pairs
	size_t nob = 4;
	double eps = 0.0001;
	size_t start_rank = 2;
	size_t max_rank = 250;
	size_t wsize = 5;
	XERUS_LOG(simpleMALS,"Load Matrix ...");
	//xerus::Tensor H = make_H(nob);
	xerus::Tensor H = make_H_BE();
  xerus::Tensor V = make_V_BE();

	std::vector<size_t> perm = cuthill_mckee(H);
	XERUS_LOG(simpleMALS,"Permutation = " << perm);
	xerus::Tensor perm_T = make_Permutation(perm, dim);
  H(i1,i2) = perm_T(i1,jj) * H(jj,kk) *perm_T(i2,kk);
  V(i1,i2,i3,i4) = perm_T(i1,ii) * perm_T(i2,jj) * perm_T(i3,kk) * perm_T(i4,ll) * V(ii,jj,kk,ll);



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


	xerus::TTTensor phi = xerus::TTTensor::random(std::vector<size_t>(dim,2),std::vector<size_t>(dim - 1,start_rank));
  xerus::TTOperator op = opH + opV;
//
//  std::string name2 = "hamiltonian" + std::to_string(d) +"_full.ttoperator";
//  std::ofstream write(name2.c_str() );
//  xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
//  write.close();
	double lambda = simpleMALS(op, phi, eps, max_rank);
  //double lambda = simpleALS(op, phi);


//	phi.round(10e-14);
	XERUS_LOG(info, "The ranks of op are " << op.ranks() );
	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	XERUS_LOG(info, "The ev error is  " << std::abs(-14.39207049621482 - lambda) );

	return 0;
}

/*
 *
 *
 *
 * Functions
 *
 *
 */

xerus::TTOperator return_op_H(xerus::Tensor H){
	size_t dim = 2*H.dimensions[0];
	auto eps = 10e-10;

	auto op = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	for (size_t i = 0; i < dim; i++){
			for (size_t j = 0; j < dim; j++){
				if (i % 2 == j % 2){
					value_t val = H[{i / 2, j / 2}];
					if (std::abs(val) < eps)
						continue;
					op += val * return_one_e_ac(i,j,dim);
				}
			}
		}
	op.round(10e-12);
	return op;
}

#if build_operator
xerus::TTOperator return_op_V(xerus::Tensor V){
	size_t dim = 2*V.dimensions[0];
	auto eps = 10e-10;
	auto op = xerus::TTOperator(std::vector<size_t>(2*dim,2));
	MPI_Init(NULL, NULL);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int len = static_cast<int>(dim) / world_size;
	int end = world_rank == world_size - 1 ? dim : world_rank * len + len;

	clock_t start_full = std::clock();
	for (size_t a = world_rank * len; a < end; a = a + 2){
		clock_t start1 = std::clock();
		for (size_t b = 0; b < dim; b = b + 2){
			clock_t start2 = std::clock();
			for (size_t c = 0; c < dim; c = c + 2){
				op.canonicalized = false;
				for (size_t d = 0; d <= c ; d = d + 2){
					value_t val = V[{a / 2, b / 2, c / 2, d / 2}];
					if (std::abs(val) < eps )//|| a == b || c: == d) // TODO check i == j || k == l
						continue;
					op += val * return_two_e_ac_full(a,b,c,d,dim);
				}
				op.move_core(0);
			}
			clock_t end2 = std::clock();
			auto elapsed_secs2 = double(end2 - start2) / CLOCKS_PER_SEC;
			XERUS_LOG(simpleALS, "Time elapsed for inner iteration(" <<world_rank << "): " << elapsed_secs2 << " sec" );
		}
		clock_t end1 = std::clock();
		auto elapsed_sec1 = double(end1 - start1) / CLOCKS_PER_SEC;
		XERUS_LOG(simpleALS,"---------------------------------------------------------------");
		XERUS_LOG(simpleALS, "Time elapsed for outer iteration (" <<world_rank << "): " << elapsed_sec1 << " sec" );

	}
	clock_t end_full = std::clock();
	auto elapsed_secs_full = double(end_full - start_full) / CLOCKS_PER_SEC;
	XERUS_LOG(simpleALS, "Time elapsed for full iteration(" <<world_rank << "): " << elapsed_secs_full << " sec" );

	std::string name2 = "hamiltonian" + std::to_string(dim) +"_" + std::to_string(world_rank) +".ttoperator";
	std::ofstream write(name2.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
	write.close();
	MPI_Finalize();
	op.round(10e-12);
	return op;
}
#endif

xerus::Tensor make_H(size_t nob){
	auto H_AO = xerus::Tensor({nob,nob});
	auto C = xerus::Tensor({nob,nob});

	std::string name = "oneParticleOperator"+std::to_string(nob)+".csv";
	Mat H_Mat = load_csv<Mat>(name);
	name = "hartreeFockEigenvectors"+std::to_string(nob)+".csv";
	Mat C_Mat = load_csv<Mat>(name);


	for(size_t i = 0; i < nob; ++i){
		for(size_t j = 0; j < nob; ++j){
			C[{i,j}] = C_Mat(i,j);
			H_AO[{i,j}] = H_Mat(i,j);
		}
	}

	xerus::Index i1,i2,ii,jj;
	auto H_MO = xerus::Tensor({nob,nob});
	H_MO(i1,i2) = C(ii,i1)*C(jj,i2)*H_AO(ii,jj);
	return H_MO;
}

xerus::Tensor make_H_BE(){
	auto H = xerus::Tensor({4,4});
	auto H2 = xerus::Tensor({8,8});
  H[{0,0}] =      -7.854546469;
  H[{1,0}] =   		0.2190509890;
  H[{0,1}] =   		0.2190509890;
  H[{1,1}] =      -1.594069613;
	H[{2,0}] =    -0.5038662979E-01;
	H[{0,2}] =    -0.5038662979E-01;
	H[{2,1}] =     0.1590245167;
	H[{1,2}] =     0.1590245167;
	H[{2,2}] =    -0.4996472436;
	H[{3,0}] =     0.2158425412;
	H[{0,3}] =     0.2158425412;
	H[{3,1}] =    -0.4395272886;
	H[{1,3}] =    -0.4395272886;
	H[{3,2}] =     0.2151398215;
	H[{2,3}] =     0.2151398215;
	H[{3,3}] =    -0.9315770207;
	for (size_t i = 0; i < 4; i++){
		for (size_t j = 0; j < 4; j++){
			auto value = H[{i,j}];
				H2[{2*i,2*j,}] = value;
				H2[{2*i+1,2*j+1}] = value;
		}
	}
	return H2;
}

xerus::Tensor make_V(size_t nob){
	auto V_AO = xerus::Tensor({nob,nob,nob,nob});
	auto C = xerus::Tensor({nob,nob});

	std::string name = "cc-pvdz_bauschlicher"+std::to_string(nob)+".tensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,V_AO,xerus::misc::FileFormat::BINARY);
	read.close();
	name = "hartreeFockEigenvectors"+std::to_string(nob)+".csv";
	Mat C_Mat = load_csv<Mat>(name);

	for(size_t i = 0; i < nob; ++i){
		for(size_t j = 0; j < nob; ++j){
			C[{i,j}] = C_Mat(i,j);
		}
	}

  xerus::Index i1,i2,i3,i4, ii,jj,kk,ll,mm;
	auto V_MO = xerus::Tensor({nob,nob,nob,nob});
  V_MO(i1,i2,i3,i4) =  V_AO(ii,jj,kk,ll) * C(ii,i1)*C(jj,i2)*C(kk,i3)*C(ll,i4);
	return V_MO;
}

xerus::Tensor make_V_BE(){
	auto V = xerus::Tensor({4,4,4,4});
	auto V_end = xerus::Tensor({4,4,4,4});
	auto V_end2 = xerus::Tensor({8,8,8,8});
	V[{0,0,0,0}] =  2.267945766    ;
	V[{1,0,0,0}] = -0.2111818790    ;
	V[{1,0,1,0}] = 0.3138258308E-01;
	V[{1,1,0,0}] = 0.4909715652;
	V[{1,1,1,0}] = -0.7869113237E-02;
	V[{1,1,1,1}] = 0.3397511542;
	V[{2,0,0,0}] = 0.4895090312E-01;
	V[{2,0,1,0}] = -0.7343246602E-02;
	V[{2,0,1,1}] = 0.1646118471E-02;
	V[{2,0,2,0}] = 0.1719084918E-02;
	V[{2,1,0,0}] = -0.6718385965E-01;
	V[{2,1,1,0}] = 0.1857595751E-02;
	V[{2,1,1,1}] = -0.3313262761E-01;
	V[{2,1,2,0}] = -0.4265413154E-03;
	V[{2,1,2,1}] = 0.6222837139E-02;
	V[{2,2,0,0}] = 0.1416413337    ;
	V[{2,2,1,0}] = -0.4337337159E-03;
	V[{2,2,1,1}] = 0.1337132305    ;
	V[{2,2,2,0}] = 0.2441914512E-04 ;
	V[{2,2,2,1}] = -0.1834373106E-02 ;
	V[{2,2,2,2}] = 0.1100227758    ;
	V[{3,0,0,0}] = -0.2072195262;
	V[{3,0,1,0}] = 0.3091786988E-01;
	V[{3,0,1,1}] = -0.8269120939E-02;
	V[{3,0,2,0}] = -0.7234672550E-02;
	V[{3,0,2,1}] = 0.1882381970E-02;
	V[{3,0,2,2}] = -0.4995136405E-03;
	V[{3,0,3,0}] = 0.3047932126E-01;
	V[{3,1,0,0}] = 0.1997162253 ;
	V[{3,1,1,0}] = -0.7920083590E-02;
	V[{3,1,1,1}] = 0.7115559280E-01;
	V[{3,1,2,0}] = 0.1832178095E-02;
	V[{3,1,2,1}] = -0.1504715160E-01;
	V[{3,1,2,2}] = 0.3567543346E-02 ;
	V[{3,1,3,0}] = -0.7669371553E-02;
	V[{3,1,3,1}] = 0.4423261061E-01;
	V[{3,2,0,0}] = -0.7453142107E-01;
	V[{3,2,1,0}] = 0.1849152148E-02;
	V[{3,2,1,1}] = -0.4417940602E-01;
	V[{3,2,2,0}] = -0.3834295785E-03 ;
	V[{3,2,2,1}] = 0.4937902093E-02 ;
	V[{3,2,2,2}] = -0.9589124471E-02;
	V[{3,2,3,0}] = 0.1861810917E-02;
	V[{3,2,3,1}] = -0.1106087118E-01;
	V[{3,2,3,2}] = 0.8838257752E-02  ;
	V[{3,3,0,0}] = 0.3890399635    ;
	V[{3,3,1,0}] = -0.7845863700E-02;
	V[{3,3,1,1}] = 0.2610468864;
	V[{3,3,2,0}] = 0.1664841315E-02;
	V[{3,3,2,1}] = -0.2047995105E-01;
	V[{3,3,2,2}] = 0.1250626594;
	V[{3,3,3,0}] = -0.7894000039E-02;
	V[{3,3,3,1}] = 0.4583223164E-01 ;
	V[{3,3,3,2}] = -0.3258593753E-01;
	V[{3,3,3,3}] = 0.2136542350 ;
	for (size_t i = 0; i < 4; i++){
			for (size_t j = 0; j <= i; j++){
				for (size_t k = 0; k<= i; k++){
					for (size_t l = 0; l <= (i==k ? j : k); l++){
						auto value = V[{i,j,k,l}];
						V_end[{i,k,j,l}] = value;
						V_end[{j,k,i,l}] = value;
						V_end[{i,l,j,k}] = value;
						V_end[{j,l,i,k}] = value;
						V_end[{k,i,l,j}] = value;
						V_end[{l,i,k,j}] = value;
						V_end[{k,j,l,i}] = value;
						V_end[{l,j,k,i}] = value;
					}
				}
			}
	}
	for (size_t i = 0; i < 4; i++){
				for (size_t j = 0; j < 4; j++){
					for (size_t k = 0; k < 4; k++){
						for (size_t l = 0; l < 4; l++){
							auto value = V_end[{i,j,k,l}];
							V_end2[{2*i,2*j,2*k,2*l}] = value;
							V_end2[{2*i+1,2*j,2*k+1,2*l}] = value;
							V_end2[{2*i,2*j+1,2*k,2*l+1}] = value;
							V_end2[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
						}
					}
				}
	}
	return V_end2;
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

class InternalSolver {
	const size_t d;
	double lambda;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x)
		: d(_x.degree()), x(_x), A(_A), maxIterations(200), lambda(1.0)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}

	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - lambda*x(i&0));
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		std::vector<double> residuals_ev(10, 1000.0);
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Eigenvalue " << std::setprecision(16) <<  lambda);

			residuals_ev.push_back(lambda);
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-5] - residuals_ev.back()) < 0.00005) {
				XERUS_LOG(simpleALS, "The residuum is " << calc_residual_norm());
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs;

				const Tensor &Ai = A.get_component(corePosition);

				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
  	  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
		  	xerus::get_smallest_eigenvalue_iterative(x.component(corePosition),op,ev.get(), 1, 50000, 1e-8);
		  	//XERUS_LOG(info,sol);
		  	lambda = ev[0];

				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
			}


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}


		}
		return lambda;
	}
};

double simpleALS(const TTOperator& _A, TTTensor& _x)  {
	InternalSolver solver(_A, _x);
	return solver.solve();
}

class InternalSolver2 {
	const size_t d;
	double lambda;
	double eps;
	size_t maxRank;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	TTTensor& x;
	const TTOperator& A;
public:
	size_t maxIterations;

	InternalSolver2(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank)
		: d(_x.degree()), x(_x), A(_A), maxIterations(200), lambda(1.0), eps(_eps), maxRank(_maxRank)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);


		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}

	double calc_residual_norm() { // TODO improve this by (A-lamdaI)
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - lambda*x(i&0));
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 1; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		Index a1, a2, a3, a4, a5, r1, r2;
		std::vector<double> residuals_ev(10, 1000.0);
		std::vector<double> residuals(10, 1000.0);
		XERUS_LOG(info,"A = " << A.ranks());
		clock_t start;
		value_t es_move_inner = 0.0;
		value_t es_contract_inner = 0.0;
		value_t es_solve_inner = 0.0;
		value_t es_svd_inner = 0.0;
		for (size_t itr = 0; itr < maxIterations; ++itr) {


			// Calculate residual and check end condition
			residuals_ev.push_back(lambda);
			//residuals.push_back(calc_residual_norm());
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-10] - residuals_ev.back()) < 0.001) {
				XERUS_LOG(simpleMALS, "Done! Residual decreased to residual "  << std::scientific << calc_residual_norm() << " in " << itr << " iterations.");
				auto total = es_move_inner+es_contract_inner+es_solve_inner + es_svd_inner;
				XERUS_LOG(info, "Inner Move:     " << es_move_inner << " \tsec " << 100*(es_move_inner)/total << " \t%");
				XERUS_LOG(info, "Inner Contract: " << es_contract_inner << " \tsec " << 100*(es_contract_inner)/total << " \t%");
				XERUS_LOG(info, "Inner Solve:    " << es_solve_inner << " \tsec " << 100*(es_solve_inner)/total << " \t%");
				XERUS_LOG(info, "Inner SVD:      " << es_svd_inner << " \tsec " << 100*(es_svd_inner)/total << " \t%");

				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d-1; ++corePosition) {

				Tensor op, rhs, U, S, Vt;
				//XERUS_LOG(simpleMALS, "Iteration: " << itr  << " core: " << corePosition  << " Eigenvalue " << std::setprecision(16) <<  lambda);

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ai1 = A.get_component(corePosition+1);

				Tensor &xi = x.component(corePosition);
				Tensor &xi1 = x.component(corePosition+1);


				auto x_rank = xi.dimensions[2];



				//XERUS_LOG(info, "Operator Size = (" << (leftAStack.back()).dimensions[0] << "x" << Ai.dimensions[1] << "x" << Ai1.dimensions[1] << "x" << rightAStack.back().dimensions[0] << ")x("<< leftAStack.back().dimensions[2] << "x" << Ai.dimensions[2] << "x" << Ai1.dimensions[2] << "x" << rightAStack.back().dimensions[2] <<")");


				Tensor sol, xright;
				//sol(a1,a2,a4,a5) = xi(a1,a2,a3)*xi1(a3,a4,a5);
				start = std::clock();
				//lambda = xerus::get_smallest_eigenvalue(sol, op);
  	  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
		  	if (xi.dimensions[2] > 32)
		  		xerus::get_smallest_eigenvalue_iterative_dmrg_special(sol,leftAStack.back(),Ai,Ai1,rightAStack.back(),ev.get(), 1, 100000, 1e-10);
		  	else {
					op(i1, i2, i3, i4, j1, j2, j3, j4) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2) * Ai1(k2,i3,j3,k3)*rightAStack.back()(i4, k3, j4);
					xerus::get_smallest_eigenvalue_iterative(sol,op,ev.get(), 1, 100000, 1e-10);
		  	}
		  	//XERUS_LOG(info,sol);
		  	lambda = ev[0];
				es_solve_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				start = std::clock();
				(U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxRank,eps);
				es_svd_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

				if (corePosition == d/2) {
					XERUS_LOG(simpleALS, "Iteration: " << itr  <<  " Eigenvalue " << std::setprecision(16) <<  lambda);// << " Residual = " << residual);
					//XERUS_LOG(info,"sigma = " << S);
				}
				xright(r1,j1,j2) = S(r1,r2)*Vt(r2,j1,j2);
				auto x_rank2 = U.dimensions[2];
				//ading random kicks
				auto xleft_kicked = xerus::Tensor({U.dimensions[0],U.dimensions[1],U.dimensions[2] + 1});
				auto random_kick = xerus::Tensor::random({U.dimensions[0],U.dimensions[1],1});
				xleft_kicked.offset_add(U,{0,0,0});
				xleft_kicked.offset_add(random_kick,{0,0,U.dimensions[2]});
				auto xright_kicked = xerus::Tensor({xright.dimensions[0] + 1,xright.dimensions[1],xright.dimensions[2]});
				xright_kicked.offset_add(xright,{0,0,0});

				//XERUS_LOG(info, "U " << U.dimensions << " Vt " << xright.dimensions);
				x.set_component(corePosition, xleft_kicked);
				x.set_component(corePosition+1, xright_kicked);\
				//XERUS_LOG(info, "After kick " << x.ranks());

//				if (x_rank < x_rank2){
//					leftAStack.clear();
//					leftAStack.emplace_back(Tensor::ones({1,1,1}));
//					for (size_t pos = 0; pos < corePosition; ++pos ){
//						push_left_stack(pos);
//					}
//				}



				if (corePosition+2 < d) {
					start = std::clock();
					x.move_core(corePosition+1, true);
					es_move_inner += double(std::clock() - start) / CLOCKS_PER_SEC;
					start = std::clock();
					push_left_stack(corePosition);
					rightAStack.pop_back();
					es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;
				}
				//XERUS_LOG(info, "After move " << x.ranks());

			}


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			start = std::clock();
			for (size_t corePosition = d-1; corePosition > 1; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}
			es_contract_inner += double(std::clock() - start) / CLOCKS_PER_SEC;

		}
		return lambda;
	}
};

double simpleMALS(const TTOperator& _A, TTTensor& _x, double _eps, size_t _maxRank)  {
	InternalSolver2 solver(_A, _x, _eps, _maxRank);
	return solver.solve();
}

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

