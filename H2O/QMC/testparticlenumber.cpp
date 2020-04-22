#include <xerus.h>

#include "../../classes/QMC/tangential_parallel.cpp"
#include "../../classes/QMC/basic.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

int main(){
	size_t nob = 24,num_elec = 8,iterations = 1e4,pos = 5, numIter = 20,rank=20;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=0.1,ev_app_tmp;
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };
	value_t nuc = -52.4190597253;
	value_t alpha = 0.1;

	auto P = particleNumberOperator(2*nob);
	auto Pup = particleNumberOperatorUp(2*nob);
	auto Pdown = particleNumberOperatorDown(2*nob);


	XERUS_LOG(info,"Loading Start vector from disc");
	TTTensor start, phi,res, phi_tmp;
	read_from_disc("../data/hf_gradient_48.tttensor",start);
	phi = makeUnitVector(sample,2*nob);

	XERUS_LOG(info,"Particle number phi start         " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi start      " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi start    " << std::setprecision(16) << contract_TT(Pdown,phi,phi));


	XERUS_LOG(info,"Round start vector to " << eps << " keeping sing values bigger than " << eps/std::sqrt(2*nob-1));
	start/= start.frob_norm();
	XERUS_LOG(info,"Particle number grad              " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up grad           " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down grad         " << std::setprecision(16) << contract_TT(Pdown,start,start));
	start.round(eps);
	start/= start.frob_norm();
	XERUS_LOG(info,"Particle number grad rounded      " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up grad rounded   " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down grad rounded " << std::setprecision(16) << contract_TT(Pdown,start,start));
	phi -= alpha*start;
	phi/=phi.frob_norm();
	phi.move_core(0);
	XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));
	XERUS_LOG(info,phi.ranks());




	return 0;
}
