#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#define build_operator 0

using namespace xerus;
using namespace Eigen;
using xerus::misc::operator<<;

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op(std::vector<value_t> coeffs);
TTOperator build_Fock_op_inv(std::vector<value_t>coeffs, size_t k);
TTOperator build_pseudo_id(size_t dim);

TTTensor   make_vector(TTOperator op);
TTOperator make_diag_operator(TTTensor vec);

class InternalSolver;
void simpleALS(const TTOperator& _A, TTTensor& _x, const TTTensor& _b);

TTOperator take_square_root(TTOperator op,size_t maxiter = 10, value_t eps = 10e-2);
TTOperator take_inv_square_root(TTOperator op,value_t start = 0.01, size_t maxrank = 10, size_t maxiter = 10, value_t eps = 10e-2);
TTOperator take_inv(TTOperator op,value_t start = 0.01, size_t maxrank = 10, size_t maxiter = 10, value_t eps = 10e-2);


value_t get_hst(size_t k);
value_t get_tj(int j, size_t k);
value_t get_wj(int j, size_t k);
value_t minimal_ev(std::vector<value_t> coeffs);


int main() {
	xerus::Index ii,jj,kk,ll;
	size_t nob = 25;
	std::string name = "hartreeFockEigenvvalues" + std::to_string(nob) +".csv";
	Mat HFev_tmp = load_csv<Mat>(name);
	size_t rank = 1;

	double shift = 86.0;

	std::vector<value_t> HFev;
	for(size_t j = 0; j < nob; ++j){
		auto val = HFev_tmp(j,0);
		HFev.emplace_back(val);
		HFev.emplace_back(val);
	}
	XERUS_LOG(info, HFev);

	XERUS_LOG(info,"Build Fock OP");
	TTOperator Fock = build_Fock_op(HFev);

	TTOperator test,pseudo_id = build_pseudo_id(2*nob);
//	test(ii/2,jj/2) = Fock(ii/2,kk/2) * Fock_inv(kk/2,ll/2) * pseudo_id(ll/2,jj/2);
//	XERUS_LOG(info, "test2 norm = " << (test - pseudo_id).frob_norm());

//	std::vector<value_t> test_Fock;
  value_t start = 0;
	for (size_t i = 0;i< 2*nob;++i){
		//test_Fock.emplace_back(i);
		start += HFev[i] + shift;
	}
	XERUS_LOG(info, "Start = " << start);

	//TTOperator Fock = build_Fock_op(test_Fock);
	//XERUS_LOG(info,"Take Square Root");
	//auto Focksq = take_square_root(Fock,10,10e-2);

	Fock += shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
	XERUS_LOG(info,"Take Square Root Inv");
	TTOperator Focksqinv = take_inv_square_root(Fock,3 / start,rank,20,3*10e-10);

//	std::vector<size_t> vec(4*nob,1);
//	vec[10] = 1;
//	vec[60] = 1;
//	vec[12] = 1;
//	vec[62] = 1;
//	XERUS_LOG(info,"Fock" << Fock[vec]);
//	XERUS_LOG(info,"Fockinv" << Focksqinv[vec]);

	name = "../data/Fock_inv_sqr_H2O_" + std::to_string(2*nob) +"rank" + std::to_string(rank) +".ttoperator";
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,Focksqinv,xerus::misc::FileFormat::BINARY);
	write.close();
//
//	TTOperator Focksq = take_square_root(Fock,50,3*10e-2);
//	name = "../data/Fock_sqr_H2O_" + std::to_string(2*nob) +".ttoperator";
//	std::ofstream  write2(name.c_str() );
//	xerus::misc::stream_writer(write2,Focksq,xerus::misc::FileFormat::BINARY);
//	write2.close();
//
//	TTOperator Fockinv = take_inv(Fock,3 / start,5,10,3*10e-1);
//	name = "../data/Fock_inv_H2O_" + std::to_string(2*nob) +".ttoperator";
//	std::ofstream  write3(name.c_str() );
//	xerus::misc::stream_writer(write3,Fockinv,xerus::misc::FileFormat::BINARY);
//	write3.close();

//	TTOperator DiagHamiltonian;
//  name = "../data/FockOperator_H2O" + std::to_string(2*nob) +"_full.ttoperator";
//	std::ifstream  read4(name.c_str() );
//	xerus::misc::stream_reader(read4,DiagHamiltonian,xerus::misc::FileFormat::BINARY);
//	read4.close();
//	xerus::TTTensor e(std::vector<size_t>(2*nob,2));
//	Tensor M;
//	for (size_t i = 0; i < 1; ++i){
//		e.component(i)[{0,0,0}] = 0.0;
//		e.component(i)[{0,1,0}] = 1.0;
//	}
//	for (size_t i = 1; i < 2*nob; ++i){
//		e.component(i)[{0,0,0}] = 1.0;
//		e.component(i)[{0,1,0}] = 0.0;
//	}
//	M() = e(ii&0)*DiagHamiltonian(ii/2,jj/2) *e(jj&0);
//	XERUS_LOG(info, "M[0] = " << M[0]);
//	DiagHamiltonian += shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
//	start = 10000;
//	M() = e(ii&0)*DiagHamiltonian(ii/2,jj/2) *e(jj&0);
//	XERUS_LOG(info, "M[0] = " << M[0]);
//	TTOperator DiagHamiltonianinv = take_inv(DiagHamiltonian,1 / start,50,20,50);
//	M() = e(ii&0)*DiagHamiltonianinv(ii/2,jj/2) *e(jj&0);
//	XERUS_LOG(info, "M[0] = " << M[0]);
//	name = "../data/DiagHamiltonian_inv_H2O_" + std::to_string(2*nob) +".ttoperator";
//	std::ofstream  write4(name.c_str() );
//	xerus::misc::stream_writer(write4,DiagHamiltonianinv,xerus::misc::FileFormat::BINARY);
//	write4.close();


	return 0;
}

TTOperator take_square_root(TTOperator op,size_t maxiter, value_t eps){
	xerus::Index ii,jj,kk,ll;
	size_t dim = op.order()/2;
	TTTensor f = make_vector(op);
	//TTOperator pseudo_id = build_pseudo_id(dim);
	TTOperator diagsq,diag = op;
	diagsq(ii/2,jj/2) = diag(ii/2,kk/2)*diag(kk/2,jj/2);
	TTTensor x = TTTensor::ones(std::vector<size_t>(dim,2));
	TTTensor b = 0.5*(make_vector(diagsq) + f);


	for(size_t j =0; j < maxiter; ++j){
		simpleALS(diag,x,b);
		//x(ii&0) = pseudo_id(ii/2,jj/2) * x(jj&0);
		diag = make_diag_operator(x);
		diagsq(ii/2,jj/2) = diag(ii/2,kk/2)*diag(kk/2,jj/2);
		b = 0.5*(make_vector(diagsq) + f);
		b.round(0.0);
		x.round(2);


		auto err = std::sqrt((diagsq-op).frob_norm()/op.frob_norm());
		XERUS_LOG(info,"error iteration " << j << ": " << err);
		if (err < eps and j > 20)
			break;
	}

	return make_diag_operator(x);
}

TTOperator take_inv_square_root(TTOperator op,value_t start, size_t maxrank, size_t maxiter, value_t eps){
	xerus::Index ii,jj,kk,ll,mm;
	size_t dim = op.order()/2;
	TTTensor check,diagcubeop,x = start * TTTensor::ones(std::vector<size_t>(dim,2));
	TTOperator id,X;
//	TTOperator pseudo_id = build_pseudo_id(dim);
//	x(ii/2,jj/2) = pseudo_id(ii/2,kk/2) * x(kk/2,jj/2);
	XERUS_LOG(info,op.ranks());
	for(size_t j =0; j < maxiter; ++j){
		X = make_diag_operator(x);

		diagcubeop(ii&0) = X(ii/2,jj/2)*x(jj&0);
		diagcubeop.round(2*maxrank);
		diagcubeop(ii&0) = X(ii/2,jj/2)*diagcubeop(jj&0);
		diagcubeop.round(2*maxrank);
		diagcubeop(ii&0) = op(ii/2,jj/2)*diagcubeop(jj&0);
		diagcubeop.round(2*maxrank);

		x = 1.5 * x - 0.5*diagcubeop;
		x.round(maxrank);
		X = make_diag_operator(x);

		check(ii&0) = op(ii/2,kk/2) *  X(kk/2,jj/2) *x(jj&0);

		auto err = std::sqrt((check - TTTensor::ones(std::vector<size_t>(dim,2))).frob_norm())/frob_norm(op);///pseudo_id.frob_norm());

		XERUS_LOG(info,"error iteration " << j << ": " << err);
		if (err < eps)
			break;
	}

	return make_diag_operator(x);
}

TTOperator take_inv(TTOperator op,value_t start, size_t maxrank, size_t maxiter, value_t eps){
	xerus::Index ii,jj,kk,ll,mm;
	size_t dim = op.order()/2;
	TTTensor check,xsqop,x = start * TTTensor::ones(std::vector<size_t>(dim,2));
	TTOperator X, id;
//	TTOperator pseudo_id = build_pseudo_id(dim);
//	x(ii/2,jj/2) = pseudo_id(ii/2,kk/2) * x(kk/2,jj/2);

	for(size_t j =0; j < maxiter; ++j){
		XERUS_LOG(info, "j = " << j);
		X = make_diag_operator(x);
		XERUS_LOG(info, "x*x");
		xsqop(ii&0) = X(ii/2,jj/2)*x(jj&0);
		XERUS_LOG(info,"round");
		xsqop.round(maxrank);
		XERUS_LOG(info, "x*a");
		xsqop(ii&0) = op(ii/2,jj/2) *xsqop(jj&0);
		XERUS_LOG(info,"round");
		xsqop.round(maxrank);
		XERUS_LOG(info, "2*x - x*x*a");
		x = 2 * x - xsqop;
		XERUS_LOG(info,"round");
		x.round(maxrank);
		XERUS_LOG(info,"ranks x = " << x.ranks());
		XERUS_LOG(info,"check");
		check(ii&0) = op(ii/2,jj/2) * x(jj&0);
		XERUS_LOG(info,"err");
		auto err = std::sqrt((check - TTTensor::ones(std::vector<size_t>(dim,2))).frob_norm());///pseudo_id.frob_norm());

		XERUS_LOG(info,"error iteration " << j << ": " << err);
		if (err < eps)
			break;
	}

	return make_diag_operator(x);
}


TTTensor make_vector(TTOperator op){
	size_t dim = op.order()/2;
	TTTensor result(std::vector<size_t>(dim,2));
	for(size_t i = 0; i < dim; ++i){
		auto tmp = op.get_component(i);
		Tensor tmp2({tmp.dimensions[0],tmp.dimensions[1],tmp.dimensions[3]});
		for (size_t j = 0; j < tmp.dimensions[0]; ++j){
			for (size_t k = 0; k < tmp.dimensions[3]; ++k){
				for (size_t l = 0; l < tmp.dimensions[1]; ++l){
					tmp2[{j,l,k}] = tmp[{j,l,l,k}];
				}
			}
		}
		result.set_component(i,tmp2);
	}
	//XERUS_LOG(info,"make vector diff " << (make_diag_operator(result)- op).frob_norm());
	result.round(0.0);

	return result;
}

TTOperator make_diag_operator(TTTensor vec){
	size_t dim = vec.order();
	TTOperator result(std::vector<size_t>(2*dim,2));
	for(size_t i = 0; i < dim; ++i){
		auto tmp = vec.get_component(i);
		Tensor tmp2({tmp.dimensions[0],tmp.dimensions[1],tmp.dimensions[1],tmp.dimensions[2]});
		for (size_t j = 0; j < tmp.dimensions[0]; ++j){
			for (size_t k = 0; k < tmp.dimensions[2]; ++k){
				for (size_t l = 0; l < tmp.dimensions[1]; ++l){
					tmp2[{j,l,l,k}] = tmp[{j,l,k}];
				}
			}
		}
		result.set_component(i,tmp2);
	}
	//XERUS_LOG(info,"make op diff " << (make_vector(result)- vec).frob_norm());
	result.round(0.0);

	//XERUS_LOG(info,"res ranks " << result.ranks());

	return result;
}

TTOperator build_pseudo_id(size_t dim){
	TTOperator result(std::vector<size_t>(2*dim,2));

	auto id = xerus::Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto aa = xerus::Tensor({1,2,2,1});
	aa[{0,0,0,0}] = 1.0;
	for (size_t comp = 0; comp < dim; ++comp){
		if (comp == 0){
				Tensor tmp = Tensor({1,2,2,2});
				tmp.offset_add(id,{0,0,0,0});
				tmp.offset_add((-1.0)*aa,{0,0,0,1});
				result.set_component(comp,tmp);
		} else if (comp == dim - 1){
			Tensor tmp = Tensor({2,2,2,1});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(aa,{1,0,0,0});
			result.set_component(comp,tmp);
		} else {
			Tensor tmp = Tensor({2,2,2,2});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(aa,{1,0,0,1});
			result.set_component(comp,tmp);
		}
	}
	return result;

}


TTOperator build_Fock_op(std::vector<value_t> coeffs){
	size_t dim = coeffs.size();

	TTOperator result(std::vector<size_t>(2*dim,2));
	size_t comp = 0;
	auto id = xerus::Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto aa = xerus::Tensor({1,2,2,1});
	aa[{0,1,1,0}] = 1.0;
	for (size_t comp = 0; comp < dim; ++comp){
		value_t coeff = coeffs[comp];
		if (comp == 0){
				Tensor tmp = Tensor({1,2,2,2});
				tmp.offset_add(id,{0,0,0,0});
				tmp.offset_add(coeff*aa,{0,0,0,1});
				result.set_component(comp,tmp);
		} else if (comp == dim - 1){
			Tensor tmp = Tensor({2,2,2,1});
			tmp.offset_add(coeff*aa,{0,0,0,0});
			tmp.offset_add(id,{1,0,0,0});
			result.set_component(comp,tmp);
		} else {
			Tensor tmp = Tensor({2,2,2,2});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(coeff*aa,{0,0,0,1});
			tmp.offset_add(id,{1,0,0,1});
			result.set_component(comp,tmp);
		}
	}
	return result;
}

TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, const size_t k){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	value_t lambda_min = -1.0 * minimal_ev(coeffs);
	TTOperator pseudo_id = build_pseudo_id(dim);
	int k_int = static_cast<int>(k);
	TTOperator inv(std::vector<size_t>(2*dim,2));
	for ( int j = -k_int; j <=k_int; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		//XERUS_LOG(info,"j = " << j << ": " << get_tj(j,k));
		for (size_t i = 0; i < dim; ++i){
			value_t coeff1 = std::exp(2*get_tj(j,k)/lambda_min*coeffs[i]);
			//XERUS_LOG(info,"coeff1 " << coeff1);

			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,1,1,0}] = coeff1;
			aa[{0,0,0,0}] = 1;
			tmp.set_component(i,aa);
		}
		//XERUS_LOG(info,"j = " << j << " k = " << k << " wjk = " << get_wj(j,k));
		//XERUS_LOG(info,"j = " << j << " k = " << k << " tjk = " << get_tj(j,k));

		value_t coeff2 = 2*get_wj(j,k)/lambda_min;
		//XERUS_LOG(info,"coeff2 " << coeff2);
		//XERUS_LOG(info,"lambda_min " << lambda_min);

		//XERUS_LOG(info,tmp.frob_norm());
		inv -= coeff2 * tmp;
		//project solution
		inv(ii/2,jj/2) = inv(ii/2,kk/2) * pseudo_id(kk/2,jj/2);
		inv.round(0.0);
	}
	return inv;
}
value_t get_hst(size_t k){
	return M_PI * M_PI / std::sqrt(static_cast<value_t>(k));
}

value_t get_tj(int j, size_t k){
	value_t hst = get_hst(k);
	return std::log(std::exp(static_cast<value_t>(j)*hst) + std::sqrt(1+std::exp(2*static_cast<value_t>(j)*hst)));
}

value_t get_wj(int j, size_t k){
	value_t hst = get_hst(k);
	return hst/std::sqrt(1+std::exp(-2*static_cast<value_t>(j)*hst));
}

value_t minimal_ev(std::vector<value_t> coeffs){
	value_t lambda;
	for (size_t i = 0; i < coeffs.size(); ++i){
		value_t coeff = coeffs[i];
		lambda += (coeff < 0 ? coeff : 0);
	}
	return lambda;
}



class InternalSolver {
	const size_t d;

	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;

	TTTensor& x;
	const TTOperator& A;
	const TTTensor& b;
	const double solutionsNorm;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x, const TTTensor& _b)
		: d(_x.order()), x(_x), A(_A), b(_b), solutionsNorm(frob_norm(_b)), maxIterations(1000)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
		leftBStack.emplace_back(Tensor::ones({1,1}));
		rightBStack.emplace_back(Tensor::ones({1,1}));
	}


	void push_left_stack(const size_t _position) {
		xerus::Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = leftBStack.back()(j1, j2)
				*xi(j1, k1, i1)*bi(j2, k1, i2);
		leftBStack.emplace_back(std::move(tmpB));
	}


	void push_right_stack(const size_t _position) {
		xerus::Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = xi(i1, k1, j1)*bi(i2, k1, j2)
				*rightBStack.back()(j1, j2);
		rightBStack.emplace_back(std::move(tmpB));
	}

	double calc_residual_norm() {
		xerus::Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - b(i&0)) / solutionsNorm;
	}


	void solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

		xerus::Index i1, i2, i3, j1 , j2, j3, k1, k2;
		std::vector<double> residuals(10, 1000.0);

		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			residuals.push_back(calc_residual_norm());
			if (residuals.back()/residuals[residuals.size()-5] > 0.99) {
				//XERUS_LOG(simpleALS, "Done! Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
				return; // We are done!
			}
			//XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back());


			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs;

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &bi = b.get_component(corePosition);

				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);
				rhs(i1, i2, i3) =            leftBStack.back()(i1, k1) *   bi(k1, i2, k2) *   rightBStack.back()(i3, k2);

				xerus::solve(x.component(corePosition), op, rhs);

				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
					rightBStack.pop_back();
				}
			}


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
				leftBStack.pop_back();
			}

		}
	}

};

void simpleALS(const TTOperator& _A, TTTensor& _x, const TTTensor& _b)  {
	InternalSolver solver(_A, _x, _b);
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
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}
