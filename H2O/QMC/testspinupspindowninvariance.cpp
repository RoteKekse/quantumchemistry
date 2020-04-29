#include <xerus.h>
#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/trialfunctions.cpp"
#include "../../classes/QMC/tangential_parallel.cpp"



int main(){
	size_t d = 48,p = 8,iterations = 1e4, rank = 10;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=1,ev_app_tmp;
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	value_t nuc = -52.4190597253;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	value_t alpha = 0.1,beta;

	xerus::TTTensor phi,res,res_last,start;
	phi = makeUnitVector(hf_sample,d);
	read_from_disc("../data/hf_gradient_48.tttensor",start);
	XERUS_LOG(info,"Round start vector to " << eps << " keeping sing values bigger than " << eps/std::sqrt(d-1));
	start/= start.frob_norm();
	for (value_t ee = 0.05; ee <= eps ; ee+=0.05){
		start.round(ee);
		start/= start.frob_norm();
	}
	auto P = particleNumberOperator(d);
	auto Pup = particleNumberOperatorUp(d);
	auto Pdown = particleNumberOperatorDown(d);

	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start));

	phi -= alpha*start;
	phi/=phi.frob_norm();
	phi.move_core(0);
	XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));
	XERUS_LOG(info,phi.ranks());

	Tangential tang(d,p,iterations,path_T,path_V,shift,hf_sample,phi);
	ev = tang.get_eigenvalue();
	XERUS_LOG(info,"Approximated Eigenvalue: " << ev- shift +nuc);

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	XERUS_LOG(info,"Exact Eigenvalue: " << contract_TT(Hs,phi,phi)- shift +nuc);



	return 0;
}
