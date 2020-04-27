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
	XERUS_LOG(info,phi.ranks());
	std::vector<size_t> hf_idx = makeIndex(hf_sample,d);

	XERUS_LOG(info, "Phi at HF sample "  << phi[hf_idx]);

	ContractionTree tree(phi,hf_sample);
	XERUS_LOG(info, "Phi at HF sample "  << tree.getValue());


	std::vector<size_t> t1 = {0,1,2,3,22,23,30,47};
	std::vector<size_t> t2 = {0,1,5,8,22,23,30,31};
	std::vector<size_t> t3 = {0,1,2,7,23,24,30,33};

	auto res1 = tree.updatedTree(t1);
	auto res2 = tree.updatedTree(t2);
	auto res3 = tree.updatedTree(t3);

	std::vector<size_t> idx1 = makeIndex(t1,d);
	std::vector<size_t> idx2 = makeIndex(t2,d);
	std::vector<size_t> idx3 = makeIndex(t3,d);

	XERUS_LOG(info, "t1 "  << phi[idx1] << " " << res1.getValue());
	XERUS_LOG(info, "t2 "  << phi[idx2] << " " << res2.getValue());
	XERUS_LOG(info, "t3 "  << phi[idx3] << " " << res3.getValue());


}
