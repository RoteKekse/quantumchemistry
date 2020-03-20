#include <xerus.h>


#include "../loading_tensors.cpp"

void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);



int main(){
	size_t nob = 24,num_elec = 8, max_rank = 20, iterations = 1e6;

	xerus::TTTensor phi,res,test;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	project(phi,num_elec,2*nob);
	std::string name = "../data/residual_" + std::to_string(2*nob)+"_"+ std::to_string(max_rank)  +"_benchmark.tttensor";
	read_from_disc(name,res);

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
  std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	value_t rHx,rHr,rx,rr,xx,xHx, alpha, t1,t2,t3;

	//phi = TTTensor::dirac(std::vector<size_t>(2*nob,2),{1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});

	res /= res.frob_norm();

	xHx = contract_TT(Hs,phi,phi);
	xx = phi.frob_norm();
	XERUS_LOG(info, "Eigenvalue exact: " << xHx / xx);


	xerus::Index i1,i2,j1,j2,k1,k2;
	t1 = contract_TT2(Hs,Hs,phi,phi);

	XERUS_LOG(info, "residuum norm " << t1 - xHx*xHx);





	rHx = contract_TT(Hs,res,phi);
	rHr = contract_TT(Hs,res,res);
	rx = contract_TT(id,res,phi);
	rr = contract_TT(id,res,res);

	//alpha = 3.0;
	alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
	XERUS_LOG(info, "Alpha: " << alpha);

	phi = phi - alpha* res;

	project(phi,num_elec,2*nob);
	XERUS_LOG(info, "Eigenvalue exact: " << contract_TT(Hs,phi,phi)/ phi.frob_norm());

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
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
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
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
	phi /= phi.frob_norm();
	XERUS_LOG(info,phi.ranks());
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
