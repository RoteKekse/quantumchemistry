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

	size_t idx = 10;
	auto split = start.chop(idx+1);

	XERUS_LOG(info, start.order());
	XERUS_LOG(info, split.first.order());
	XERUS_LOG(info, split.second.order());

	TTTensor start_first(std::vector<size_t>(idx,2));
	for (size_t i = 0; i < idx-1; ++i){
		Tensor tensor(*split.first.nodes[i+1].tensorObject);
		XERUS_LOG(info,tensor.dimensions);
		start_first.set_component(i,tensor);
	}
	Tensor tensor(*split.first.nodes[idx].tensorObject);

	for (size_t slice = 0; slice < 10; slice++){
		tensor.fix_mode(2,slice);
		XERUS_LOG(info,tensor.dimensions);
		tensor.reinterpret_dimensions({tensor.dimensions[0],tensor.dimensions[1],1});
		start_first.set_component(idx-1,tensor);

		auto Psplit = particleNumberOperator(idx);
		auto idsplit = xerus::TTOperator::identity(std::vector<size_t>(2*idx,2));

		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplit,start_first,start_first)/contract_TT(idsplit,start_first,start_first));
	}

	return 0;
}
