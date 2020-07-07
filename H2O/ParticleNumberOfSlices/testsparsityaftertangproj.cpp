#include <xerus.h>
#include <chrono>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"
#include "../../classes/GradientMethods/ALSres.cpp"




int main(){
	size_t d = 48,p = 8,iterations = 1e6,iterations2 = 100*iterations,roundIter = 10, rank = 10;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=1.0,ev_app_tmp;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	value_t alpha = 0.1,beta;

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(2*d,2));
	auto P = particleNumberOperator(d);
	auto Pup = particleNumberOperatorUp(d);
	auto Pdown = particleNumberOperatorDown(d);

	xerus::TTTensor phi,res,res_last,start,start2,start3,start4;
	phi = makeUnitVector(hf_sample,d);
	read_from_disc("../data/hf_gradient_48.tttensor",start);

	start/= start.frob_norm();
	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start));
	XERUS_LOG(info,start.ranks());

	size_t comp = 3;
	start.move_core(comp);
	XERUS_LOG(info, start.component(comp).dimensions);
	XERUS_LOG(info,  "\n" << start.component(comp));
	TangentialOperation top(start);

	auto tang_ex = top.localProduct(start,Hs);
	XERUS_LOG(info, tang_ex[comp].dimensions);
	XERUS_LOG(info, "\n" << tang_ex[comp]);

	return 0;
}
