#include "../classes/QMC/contractpsihek.cpp"

#include <xerus.h>
#include <../classes/helpers.cpp>

using namespace xerus;
using xerus::misc::operator<<;


int main(){

	size_t rank = 5,d = 48, p = 8;

	TTTensor phi = TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d-1,rank));





	return 0;
}
