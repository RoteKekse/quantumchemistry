#include <xerus.h>
#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/trialfunctions.cpp"

std::pair<size_t,size_t> getUPDown(std::vector<size_t> s){
	size_t pup=0, pdown=0;
	for (size_t i : s){
		if (i%2 == 0)
			pup++;
		else
			pdown++;
	}
	return std::pair<size_t,size_t>(pup,pdown);
}

int main(){
	size_t d = 48,p = 8,iterations = 1e5, rank = 10;
	value_t ev, shift = 25.0, ev_app;

	TTTensor direction;
	read_from_disc("../data/hf_gradient_48.tttensor",direction);
	direction /= direction.frob_norm();
	direction.round(10);

	auto id = xerus::TTOperator::identity(std::vector<size_t>(2*d,2));
	XERUS_LOG(info,"Direction norm   " <<direction.frob_norm());
	XERUS_LOG(info,"Direction ranks  " <<direction.ranks());

	std::vector<size_t> sample({0,1,2,3,22,23,30,31});
	auto ehf = makeUnitVector(sample,d);
	while(true){
		auto ek = makeUnitVector(sample,d);
		value_t val = contract_TT(id,ek,direction);
		if (std::abs(val)> 1e-12)
			XERUS_LOG(info,getUPDown( sample) << " \t" << sample<<": \t" <<contract_TT(id,ek,direction));
		sample = TrialSample(sample,d);

	}


//
//	std::pair<std::vector<size_t>,std::vector<size_t>> sample({0,1,11,15},{0,1,11,15});
//	auto ehf = makeUnitVector(sample,d);
//	while(true){
//		auto ek = makeUnitVector(sample,d);
//		value_t val = contract_TT(id,ek,direction);
//		if (std::abs(val)> 1e-12)
//			XERUS_LOG(info,sample<<": \t" <<contract_TT(id,ek,direction));
//		sample = TrialSample(sample,d);
//
//	}






	return 0;
}
