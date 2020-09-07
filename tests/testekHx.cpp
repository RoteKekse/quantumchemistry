#include "../classes/QMC/contractpsihek.cpp"

#include <xerus.h>
#include <../classes/helpers.cpp>

using namespace xerus;
using xerus::misc::operator<<;


int main(){

	size_t rank = 5,d = 48, p = 8;

	TTTensor phi = TTTensor::random(std::vector<size_t>(2,d),std::vector<size_t>(rank,d-1));





	return 0;
}
