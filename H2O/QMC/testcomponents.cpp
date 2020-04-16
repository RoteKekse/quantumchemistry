#include <xerus.h>

#include "../../classes/QMC/tangential_parallel.cpp"
#include "../../classes/QMC/basic.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"


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


	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };
	Tangential tang(2*nob,num_elec,iterations,path_T,path_V,shift,sample,phi);
	tang.uvP.xbase.first = top.xbasis[0]; //use same orthogonalization!!!
	tang.uvP.xbase.second = top.xbasis[1];


	ev_app = tang.get_eigenvalue();
	XERUS_LOG(info, "Eigenvalue approx. " << ev_app);

	auto tang_app = tang.get_tangential_components(ev_app,0.001);


	res = top.builtTTTensor(tang_app);
	name2 = "../data/new_direction 	.tttensor";
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

