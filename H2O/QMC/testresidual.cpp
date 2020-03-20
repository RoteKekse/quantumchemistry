#include <xerus.h>
#include "classes/tangentialOperation.cpp"

#include "../loading_tensors.cpp"

void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
TTOperator particleNumberOperator(size_t d);
double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);



int main(){
	size_t nob = 24,num_elec = 8, max_rank = 20, iterations = 1e6;
	auto P = particleNumberOperator( 2*nob);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	xerus::TTOperator Hs, Fock_inv;
	xerus::TTTensor phi1,phi2,phi3,phi,res_app,res_ex,test, res_proj;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	project(phi,num_elec,2*nob);
	project(phi,num_elec,2*nob);
	phi.round(1e-8);
	XERUS_LOG(info, "Number exact phi: " << std::setprecision(13)<< contract_TT(P,phi,phi)/contract_TT(id,phi,phi));

	XERUS_LOG(info,phi.ranks());
	read_from_disc("../data/fock_h2o_inv_shifted48_full.ttoperator",Fock_inv);
	Fock_inv = -1*Fock_inv;
	XERUS_LOG(info,Fock_inv.ranks());


	std::string name = "../data/residual_app48_benchmark_diag3_next_eigenvector_H2O_48_3_-23.647510_benchmark.tttensor";
	read_from_disc(name,res_app);

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	TangentialOperation top(phi);
	value_t rHx,rHr,rx,rr,xx,xHx, alpha, t1,t2,t3;

	xHx = contract_TT(Hs,phi,phi);
	xx = phi.frob_norm();

	auto tang_ex = top.localProduct(Hs,Fock_inv,xHx/xx,true);
	res_ex = top.builtTTTensor(tang_ex);
	XERUS_LOG(info,"Norm res_ex " << res_ex.frob_norm());

	res_ex /= res_ex.frob_norm();



	res_app /= res_app.frob_norm();

	xHx = contract_TT(Hs,phi,phi);
	xx = phi.frob_norm();
	XERUS_LOG(info, "Eigenvalue exact: " << xHx / xx);




	XERUS_LOG(info,"-------------------------------------- approximated optimal step size");


	rHx = contract_TT(Hs,res_app,phi);
	rHr = contract_TT(Hs,res_app,res_app);
	rx = contract_TT(id,res_app,phi);
	rr = contract_TT(id,res_app,res_app);

	//alpha = 0.1;
	alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
	XERUS_LOG(info, "rHx: " << rHx);
	XERUS_LOG(info, "rHr: " << rHr);
	XERUS_LOG(info, "rx: " << rx);
	XERUS_LOG(info, "rr: " << rr);
	XERUS_LOG(info, "xx: " << xx);
	XERUS_LOG(info, "xHx: " << xHx);
	XERUS_LOG(info, "Alpha: " << alpha);

	phi1 = phi - alpha* res_app;

	XERUS_LOG(info, "Eigenvalue app: " << contract_TT(Hs,phi1,phi1)/ contract_TT(id,phi1,phi1));
	XERUS_LOG(info, "Number app: " << std::setprecision(13) <<contract_TT(P,phi1,phi1)/ contract_TT(id,phi1,phi1));
	XERUS_LOG(info,phi1.ranks());
	write_to_disc("../data/residual_app_" + std::to_string(2*nob)  +"_benchmark_diag_two_step.tttensor",phi1);

	XERUS_LOG(info,"-------------------------------------- exact optimal step size");


	rHx = contract_TT(Hs,res_ex,phi);
	rHr = contract_TT(Hs,res_ex,res_ex);
	rx = contract_TT(id,res_ex,phi);
	rr = contract_TT(id,res_ex,res_ex);
	XERUS_LOG(info, "rHx: " << rHx);
	XERUS_LOG(info, "rHr: " << rHr);
	XERUS_LOG(info, "rx: " << rx);
	XERUS_LOG(info, "rr: " << rr);
	XERUS_LOG(info, "xx: " << xx);
	XERUS_LOG(info, "xHx: " << xHx);


	//alpha = 0.1;
	alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
	XERUS_LOG(info, "Alpha: " << alpha);

	phi2 = phi - alpha* res_ex;
	XERUS_LOG(info, "Eigenvalue exact: " << contract_TT(Hs,phi2,phi2)/ contract_TT(id,phi2,phi2));
	XERUS_LOG(info, "Number exact: " << std::setprecision(13)<< contract_TT(P,phi2,phi2)/contract_TT(id,phi2,phi2));
	write_to_disc("../data/residual_ex_" + std::to_string(2*nob)  +"_benchmark_diag_two_step.tttensor",phi1);




	XERUS_LOG(info, "Error 1-2 " << (phi1-phi2).frob_norm());
	return 0;
}

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx){
	double a = rFr*rHx-rHr*rFx;
	double b = rHr*xFx-rFr*xHx;
	double c = rFx*xHx-rHx*xFx;

	double disc = b*b-4*a*c;
	double alpha1 = (-b + std::sqrt(disc))/(2*a);
	double alpha2 = (-b - std::sqrt(disc))/(2*a);
	return alpha1;
}


void project(TTTensor &phi, size_t p, size_t d){
	Index i1,i2,j1,j2,k1,k2;
	for (size_t k = 0; k <= d; ++k){
		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t)p - (value_t) k;
			phi /=  f;
			phi.round(1e-12);
		}
	}
	phi.round(1e-4);
	phi /= phi.frob_norm();
}



TTOperator particleNumberOperator(size_t k, size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;

	value_t kk = (value_t) k;
	auto kkk = kk/(value_t) d * id;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n-kkk,{0,0,0,1});
	op.set_component(0,tmp);

	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n-kkk,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n-kkk,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
}

TTOperator particleNumberOperator(size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n,{0,0,0,1});
	op.set_component(0,tmp);
	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
}
