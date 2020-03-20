#include <xerus.h>
#include <Eigen/Dense>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include "ALSres.cpp"



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

xerus::TTOperator return_op_H(size_t dim);
xerus::TTOperator return_op_V(size_t dim);



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


	size_t d = 12; // 16 electron, 8 electron pairs
	size_t nob = 6;
	double eps = 10e-6;
	size_t start_rank = 2;
	size_t max_rank =100;
	size_t wsize = 5;
	XERUS_LOG(simpleMALS,"Load Matrix ...");




	xerus::TTOperator op;
	op = return_op_H(nob) + return_op_V(nob);


	xerus::TTTensor phi = xerus::TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d - 1,start_rank));

	double lambda = simpleMALS(op, phi, eps, max_rank);
  //double lambda = simpleALS(op, phi);



	phi.round(10e-14);
	XERUS_LOG(info, "The ranks of op are " << op.ranks() );
	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "lambda op = " << lambda  );

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

xerus::TTOperator return_op_H(size_t dim){
	auto oneelec_tmp = xerus::Tensor({dim,dim});
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0,1.0);

	for (size_t i = 0; i < dim; ++i){
	    for (size_t j = 0; j < i+1; ++j){
	        value_t value = distribution(generator);
	        oneelec_tmp[{i,j}] = value;
	        oneelec_tmp[{j,i}] = value;
	    }
	}
	auto oneelec = xerus::Tensor({2*dim,2*dim});
	for (size_t i = 0; i < dim; ++i){
		    for (size_t j = 0; j < dim; ++j){
		        value_t value = oneelec_tmp[{i,j}];
		        oneelec[{2*i,2*j}] = value;
						oneelec[{2*i+1,2*j+1}] = value;
						oneelec[{2*j,2*i}] = value;
						oneelec[{2*j+1,2*i+1}] = value;
		    }
		}
	auto op = xerus::TTOperator(std::vector<size_t>(4*dim,2));
	for (size_t i = 0; i < 2*dim; i++){
			for (size_t j = 0; j < 2*dim; j++){
				if (i % 2 == j % 2){
					value_t val = oneelec[{i / 2, j / 2}];
					op += val * return_one_e_ac(i,j,2*dim);
				}
			}
		}
	op.round(10e-12);
	return op;
}

xerus::TTOperator return_op_V(size_t dim){
	auto twoelec_tmp = xerus::Tensor({dim,dim,dim,dim});
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0,1.0);

  for (size_t i = 0; i < dim; ++i) {
      for (size_t j = 0; j < i + 1; ++j){
          for (size_t k = 0; k < i + 1; ++k){
          	for (size_t l = 0; l < (i==k ? j+1 : k+1);++l){
							value_t value = distribution(generator);
							twoelec_tmp[{i,k,j,l}] = value;
							twoelec_tmp[{j,k,i,l}] = value;
							twoelec_tmp[{i,l,j,k}] = value;
							twoelec_tmp[{j,l,i,k}] = value;
							twoelec_tmp[{k,i,l,j}] = value;
							twoelec_tmp[{l,i,k,j}] = value;
							twoelec_tmp[{k,j,l,i}] = value;
							twoelec_tmp[{l,j,k,i}] = value;
          	}
          }
	    }
	}
	auto twoelec = xerus::Tensor({2*dim,2*dim,2*dim,2*dim});
	for (size_t i = 0; i < dim; ++i){
		for (size_t j = 0; j < dim; ++j){
			for (size_t k = 0; k < dim; ++k){
				for (size_t l = 0; l < dim; ++l){
		        value_t value = twoelec_tmp[{i,j,k,l}];
		        twoelec[{2*i,2*j,2*k,2*l}] = value;
						twoelec[{2*i+1,2*j,2*k+1,2*l}] = value;
						twoelec[{2*i,2*j+1,2*k,2*l+1}] = value;
						twoelec[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
				}
			}
		}
	}
	auto op = xerus::TTOperator(std::vector<size_t>(4*dim,2));
	for (size_t i = 0; i < 2*dim; ++i){
		XERUS_LOG(info,i);
			for (size_t j = 0; j < 2*dim; ++j){
				for (size_t k = 0; k < 2*dim; ++k){
					for (size_t l = 0; l < 2*dim; ++l){
	                    value_t val = twoelec[{i,j,k,l}];
	                    if (std::abs(val) < 10e-13 || (i%2 != k%2) || (j%2!=l%2))
	                        continue;
	                    op += 0.5*val * return_two_e_ac(i,j,l,k,2*dim);
					}
				}
			}
	}
	op.round(10e-12);
	return op;
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
		Index ii,jj,kk,ll,mm,nn,oo,i1,i2,i3,i4;
		auto ones = Tensor::ones({1,1,1});
		xerus::Tensor tmp = ones;
		XERUS_LOG(info,"lambda = " << lambda);

		for (size_t i = 0; i < d; i++){
			auto Ai = A.get_component(i);
			auto xi = x.get_component(i);
			tmp(i1,i2,i3) = tmp(ii,jj,kk) * xi(ii,ll,i1) * Ai(jj,ll,mm,i2) * xi(kk,mm,i3);
		}
		tmp() = tmp(ii,jj,kk) * ones(ii,jj,kk);
		XERUS_LOG(info,"xAx = " << tmp);

		auto ones2 = Tensor::ones({1,1,1,1});
		xerus::Tensor tmp2 = ones2;
		for (size_t i = 0; i < d; i++){
			auto Ai = A.get_component(i);
			auto xi = x.get_component(i);
			//XERUS_LOG(info,i);
			//XERUS_LOG(info,tmp2.dimensions);
			tmp2(i1,i2,i3,i4) = tmp2(ii,jj,kk,ll) * xi(ii,mm,i1) * Ai(jj,mm,nn,i2) * Ai(kk,nn,oo,i3) * xi(ll,oo,i4);
		}
		tmp2() = tmp2(ii,jj,kk,ll) * ones2(ii,jj,kk,ll);

	//	XERUS_LOG(info,"xAAx = " << tmp2);

		xerus::TTTensor tmp3;
//		tmp3(ii&0) = A(ii/2,jj/2) * x(jj&0);
//		XERUS_LOG(info,tmp3.ranks());
		return std::sqrt(std::abs(tmp2[0]-lambda*lambda));
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

		for (size_t itr = 0; itr < maxIterations; ++itr) {


			// Calculate residual and check end condition
			residuals_ev.push_back(lambda);
			//residuals.push_back(calc_residual_norm());
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-10] - residuals_ev.back()) < eps) {
				XERUS_LOG(info, eps);
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
				sol(a1,a2,a4,a5) = xi(a1,a2,a3)*xi1(a3,a4,a5);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
  	  	std::unique_ptr<double[]> ev(new double[1]);      // real eigenvalues
		  	if (xi.dimensions[2] > 32)
		  		xerus::get_smallest_eigenvalue_iterative_dmrg_special(sol,leftAStack.back(),Ai,Ai1,rightAStack.back(),ev.get(), 1, 100000, 10e-7);
		  	else {
					op(i1, i2, i3, i4, j1, j2, j3, j4) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2) * Ai1(k2,i3,j3,k3)*rightAStack.back()(i4, k3, j4);
					xerus::get_smallest_eigenvalue_iterative(sol,op,ev.get(), 1, 100000, 10e-7);
		  	}
		  	//XERUS_LOG(info,sol);
		  	lambda = ev[0];

				(U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxRank,eps);



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
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
				//XERUS_LOG(info, "After move " << x.ranks());

			}
			auto ranks = x.ranks();
//				std::transform(ranks.begin(), ranks.end(), ranks.begin(), std::bind(std::multiplies<size_t>(), std::placeholders::_1, 1));
//				TTTensor res = TTTensor::random(x.dimensions,ranks);
//				getRes(A, x,lambda, res);
			XERUS_LOG(simpleALS, "Iteration: " << itr  <<  " EV " << std::setprecision(16) << lambda << " res = " << calc_residual_norm());// << " Residual = " << calc_residual_norm());
			XERUS_LOG(info,"x = " << x.ranks());


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 1; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}

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

