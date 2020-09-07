#include "../classes/QMC/contractpsihek.cpp"

#include <xerus.h>
#include <../classes/helpers.cpp>

using namespace xerus;
using xerus::misc::operator<<;


int main(){
	std::string path_T = "../H2O/data/T_H2O_48_bench_single.tensor";
	std::string path_V = "../H2O/data/V_H2O_48_bench_single.tensor";
	size_t rank = 5,d = 48, p = 8;
	value_t shift = 25;

	TTTensor phi = TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d-1,rank));

	ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift);




	return 0;
}
