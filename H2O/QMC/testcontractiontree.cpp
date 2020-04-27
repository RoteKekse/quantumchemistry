#include <xerus.h>
#include "../../classes/QMC/contractiontree.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"


using namespace xerus;
using xerus::misc::operator<<;

int main(){
	size_t d = 48, p = 8;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	std::vector<size_t> hf_idx = makeIndex(hf_sample,d);

	XERUS_LOG(info, "Phi at HF sample "  << phi[hf_idx]);

	ContractionTree tree(phi,hf_sample);
	XERUS_LOG(info, "Phi at HF sample "  << tree.getValue());

}
