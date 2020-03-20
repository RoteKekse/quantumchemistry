#include <xerus.h>
#include "ALSres.cpp"

using namespace xerus;
using xerus::misc::operator<<;

double get_stepsize(double lambda,double a2, double a3, double b2);
TTOperator build_Fock_op(std::vector<value_t> coeffs);
TTOperator take_inv_square_root(TTOperator op,value_t start = 0.01, size_t maxrank = 10, size_t maxiter = 10, value_t eps = 10e-2);
TTOperator make_diag_operator(TTTensor vec);


int main(){
	size_t max_iter = 20;
	size_t max_rank = 10;
	Index ii,jj,kk,ll,mm;
	size_t nob = 3;
	size_t shift = 1.0;
	double alpha;

	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi(std::vector<size_t>(nob,2));
	xerus::TTTensor phi2(std::vector<size_t>(nob,2));
	xerus::TTTensor phi3(std::vector<size_t>(nob,2));
	xerus::TTOperator id = xerus::TTOperator::identity(std::vector<size_t>(2*nob,2));
	TTOperator Mi3,Misr, Hs,Hsi;

	TTOperator F = build_Fock_op({1,2,3}) + shift * TTOperator::identity(std::vector<size_t>(2*nob,2));
	Hs = build_Fock_op({-0.5,1,2}) + shift * TTOperator::identity(std::vector<size_t>(2*nob,2));
	Misr =  take_inv_square_root(F, 0.01,1,20,0.7);
	Mi3(ii/2,jj/2) = Misr(ii/2,kk/2) *  Misr(kk/2,jj/2);
	Hsi(ii/2,jj/2) = Misr(ii/2,kk/2) * Hs(kk/2,ll/2) * Misr(ll/2,jj/2);
	Tensor FT(F), FTisr(Misr), HsT(Hs), HsiT(Hsi), Mi3T(Mi3);
	XERUS_LOG(info,FT);
	XERUS_LOG(info,FTisr);
	XERUS_LOG(info,HsT);
	XERUS_LOG(info,HsiT);
	XERUS_LOG(info,Mi3T);


	phi = xerus::TTTensor::random(std::vector<size_t>(nob,2),std::vector<size_t>(nob-1,1));
	XERUS_LOG(info, "Multiplying Start Vector by square root of Fock Operator");
	//phi /= phi.frob_norm(); //normalize



//	phi.canonicalized = false;
//	for (size_t i =0; i < 3; ++i){
//		phi.component(i)[{0,0,0}] = 0.0;
//		phi.component(i)[{0,1,0}] = 1.0;
//	}
//	for (size_t i =0; i < 3; ++i){
//		phi.component(i)[{0,0,0}] = 1.0;
//		phi.component(i)[{0,1,0}] = 0.0;
//	}
	phi2 = phi;
	phi3 = phi;
	XERUS_LOG(info,"--- starting gradient descent ---");
	Tensor tmp,rHx,rHr,rx;
	TTTensor tmp1;
	tmp()= phi(ii&0) * Hs(ii/2,jj/2) * phi(jj&0); //get ritz value
	for (size_t iter = 0; iter < max_iter; ++iter){
		TTTensor res = TTTensor::random(std::vector<size_t>(nob, 2), {max_rank});
		//update phi
		XERUS_LOG(info, "------ Iteration = " << iter);
//		XERUS_LOG(info,"Gradient step without PC");
//		phi2 /= phi2.frob_norm(); //normalize
//		tmp()= phi2(ii&0) * Hs(ii/2,jj/2) * phi2(jj&0); //get ritz value
//		XERUS_LOG(info, "lambda = " << tmp[0]  -shift);
//		getRes(Hs,phi2,id,tmp[0],res);
//		res /= res.frob_norm();
//		rHx()= res(ii&0) * Hs(ii/2,jj/2) * phi2(jj&0); //get ritz value
//		rHr()= res(ii&0) * Hs(ii/2,jj/2) * res(jj&0); //get ritz value
//		rx()= res(ii&0) * phi2(ii&0); //get ritz value
//		alpha = get_stepsize(tmp[0],rHx[0],rHr[0],rx[0]);
//		phi2 = phi2 - alpha * res;
//		phi2.round(max_rank); //round phi
//
//		XERUS_LOG(info,"Gradient step with PC (non symmetric)");
//		phi /= phi.frob_norm(); //normalize
//		tmp()= phi(ii&0) * Hs(ii/2,jj/2) * phi(jj&0); //get ritz value
//		XERUS_LOG(info, "lambda = " << tmp[0]  -shift);
//		getRes(Hs,phi,id,tmp[0],res);
//		res(ii&0) = Mi3(ii/2,jj/2) * res(jj&0);
//		res /= res.frob_norm();
//		rHx()= res(ii&0) * Hs(ii/2,jj/2) * phi(jj&0); //get ritz value
//		rHr()= res(ii&0) * Hs(ii/2,jj/2) * res(jj&0); //get ritz value
//		rx()= res(ii&0) * phi(ii&0); //get ritz value
//		alpha = get_stepsize(tmp[0],rHx[0],rHr[0],rx[0]);
//		phi = phi - alpha* res;
//		phi.round(max_rank); //round phi

		XERUS_LOG(info,"Gradient step with PC (symmetric)");
		Tensor phi3T(phi3);
		//XERUS_LOG(info,phi3T);
		tmp()= phi3(ii&0) * Mi3(ii/2,jj/2) * phi3(jj&0); //get ritz value
		phi3 /= std::sqrt(tmp[0]);//normalize
		Tensor phi32T(phi3);
		//XERUS_LOG(info,phi32T);
		tmp()= phi3(ii&0) * Hsi(ii/2,jj/2) * phi3(jj&0); //get ritz value
		XERUS_LOG(info, "lambda = " << tmp[0]);
		getRes(Hsi,phi3,Mi3,tmp[0],res);
		tmp()= res(ii&0) * Mi3(ii/2,jj/2) * res(jj&0); //get ritz value
		res /= std::sqrt(tmp[0]);
		rHx()= res(ii&0) * Hsi(ii/2,jj/2) * phi3(jj&0); //get ritz value
		rHr()= res(ii&0) * Hsi(ii/2,jj/2) * res(jj&0); //get ritz value
		rx()= res(ii&0) * Mi3(ii/2,jj/2) * phi3(jj&0); //get ritz value
		alpha = get_stepsize(tmp[0],rHx[0],rHr[0],rx[0]);
		phi3 = phi3 - alpha* res;
		phi3.round(max_rank); //round phi

	}


	return 0;
}

double get_stepsize(double lambda,double a2, double a3, double b2){
	double a = a2 - a3 * b2;
	double b = a3 -lambda*b2;
	double c = lambda*b2 -a2;

	double disc = b*b-4*a*c;
	double alpha1 = (-b + std::sqrt(disc))/(2*a);
	double alpha2 = (-b - std::sqrt(disc))/(2*a);
	XERUS_LOG(info,"alpha1 = "<< alpha1);
	XERUS_LOG(info,"alpha2 = "<< alpha2);
	return alpha1;
}


TTOperator take_inv_square_root(TTOperator op,value_t start, size_t maxrank, size_t maxiter, value_t eps){
	xerus::Index ii,jj,kk,ll,mm;
	size_t dim = op.order()/2;
	TTTensor check,diagcubeop,x = start * TTTensor::ones(std::vector<size_t>(dim,2));
	TTOperator id,X;
//	TTOperator pseudo_id = build_pseudo_id(dim);
//	x(ii/2,jj/2) = pseudo_id(ii/2,kk/2) * x(kk/2,jj/2);

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

		auto err = std::sqrt((check - TTTensor::ones(std::vector<size_t>(dim,2))).frob_norm());///pseudo_id.frob_norm());

		XERUS_LOG(info,"error iteration " << j << ": " << err);
		if (err < eps)
			break;
	}

	return make_diag_operator(x);
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

