#include <xerus.h>

#include "ttcontainer.h"


using namespace xerus;
using xerus::misc::operator<<;

int main(){

	TTContainer ttc(20,2,5);
	XERUS_LOG(info, "ranks: " << ttc.get_container().ranks());
	XERUS_LOG(info, "capacity: " << ttc.capacity);
	XERUS_LOG(info, "capacity: " << ttc.capacity_used);

	TTTensor x = TTTensor::random(std::vector<size_t>(5,2),std::vector<size_t>(4,2));

	ttc.addTT(x,5);
	XERUS_LOG(info, "diff: " << (ttc.get_container()-x).frob_norm());
	XERUS_LOG(info, "diff: " << (x).frob_norm());
	XERUS_LOG(info, "diff: " << (ttc.get_container()).frob_norm());

	XERUS_LOG(info, "ranks: " << ttc.get_container().dimensions);
	XERUS_LOG(info, "1st comp: " << ttc.get_container().get_component(4));
	//ttc.addTT(x,3);
	//XERUS_LOG(info, "1st comp: " << ttc.get_container().get_component(1));

	return 0;

}
