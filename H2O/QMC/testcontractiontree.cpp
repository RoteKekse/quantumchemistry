#include <xerus.h>
#include "../../classes/QMC/contractiontree.cpp"

#include "../../classes/loading_tensors.cpp"


using namespace xerus;
using xerus::misc::operator<<;

int main(){
	size_t d = 48, p = 8;
	std::vector< size_t> hf_sample = {01,2,3,22,23,30,21};
	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	ContractionTree(phi,hf_sample);
}
