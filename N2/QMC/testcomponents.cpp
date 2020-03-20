#include <xerus.h>

#include "classes/tangential_parallel.cpp"
#include "classes/basic.cpp"
#include "classes/tangentialOperation.cpp"

#include "../loading_tensors.cpp"

TTOperator particleNumberOperator(size_t k, size_t d);
void project(TTTensor &phi, size_t p, size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);


int main(){
	size_t nob = 24,num_elec = 8,iterations = 1e5,pos = 5;
	value_t ev, shift = 25.0, ev_app;

	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	//read_from_disc("../data/residual_app_" + std::to_string(2*nob)  +"_benchmark_diag_one_step.tttensor",phi);
	phi /= phi.frob_norm(); //normalize
	project(phi,num_elec,2*nob);

	xerus::TTOperator Hs, Fock, Fock_inv;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));
	name2 = "../data/fock_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Fock);
	name2 = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Fock_inv);



	TangentialOperation top(phi);
	ev = contract_TT(Hs,phi,phi);
	auto tang_ex = top.localProduct(Hs,Fock_inv,ev,true);
	XERUS_LOG(info, "Eigenvalue exact " << ev);


	std::string path_T = "../data/T_H2O_48_bench.tensor";
	std::string path_V= "../data/V_H2O_48_bench.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };
	Tangential tang(2*nob,num_elec,iterations,path_T,path_V,shift,sample,phi);
	tang.uvP.xbase.first = top.xbasis[0]; //use same orthogonalization!!!
	tang.uvP.xbase.second = top.xbasis[1];


	ev_app = tang.get_eigenvalue();
	XERUS_LOG(info, "Eigenvalue approx. " << ev_app);

	auto tang_app = tang.get_tangential_components(ev_app,0.005);


	res = top.builtTTTensor(tang_app);
	name2 = "../data/residual_app" + std::to_string(2*nob) +"_benchmark_diag3_next_eigenvector_H2O_48_3_-23.647510_benchmark.tttensor";
	write_to_disc(name2,res);



	XERUS_LOG(info,"Ev error     "<<std::abs(ev - ev_app));

	XERUS_LOG(info, "Position " << 5);
	XERUS_LOG(info,"Exact component:  "<< tang_ex[5].frob_norm() << "\n" << tang_ex[5]);
	XERUS_LOG(info,"Approx component: "<< tang_app[5].frob_norm() << "\n" << tang_app[5]);
	XERUS_LOG(info,"Error (rel):    "<< (tang_ex[5]-tang_app[5]).frob_norm()/tang_ex[5].frob_norm() );
//
//	XERUS_LOG(info, "Position " << pos+1);
//	XERUS_LOG(info,"Exact component:  "<< tang_ex[pos+1].frob_norm() << "\n" << tang_ex[pos+1]);
//	XERUS_LOG(info,"Approx component: "<< tang_app[1].frob_norm() << "\n" << tang_app[1]);
//	XERUS_LOG(info,"Error (rel):    "<< (tang_ex[pos+1]-tang_app[1]).frob_norm()/ tang_ex[pos+1].frob_norm() );



	return 0;
}

TTTensor makeUnitVector(std::vector<size_t> sample, size_t d){
	std::vector<size_t> index(d, 0);
	for (size_t i : sample)
		if (i < d)
			index[i] = 1;
	auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
	return unit;
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



