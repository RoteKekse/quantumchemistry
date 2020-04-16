#include <xerus.h>
#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"



int main(){
	size_t d = 48,p = 8,iterations = 1e5, rank = 10;
	value_t ev, shift = 25.0, ev_app;

	TTTensor direction;
	read_from_disc("../data/hf_gradient_48.tttensor",direction);
	direction.round(0.95);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(2*d,2));
	XERUS_LOG(info,"Direction norm   " <<direction.frob_norm());
	XERUS_LOG(info,"Direction ranks  " <<direction.ranks());


	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	auto ehf = makeUnitVector(hf_sample,d);


	std::vector<size_t> test_sample = {0,1,2,4,22,23,30,31};
	auto etest = makeUnitVector(test_sample,d);


	XERUS_LOG(info,"HFsample, direction   " <<contract_TT(id,ehf,direction));
	XERUS_LOG(info,"TestSample, direction " <<contract_TT(id,etest,direction));




	return 0;
}
