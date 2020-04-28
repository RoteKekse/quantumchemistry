#include <xerus.h>
#include <chrono>

#include "../../classes/QMC/contractiontree.cpp"
#include "../../classes/QMC/trialfunctions.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"


using namespace xerus;
using xerus::misc::operator<<;

int main(){
	size_t d = 48, p = 8,test_number = 1e5;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	XERUS_LOG(info,phi.ranks());
	std::vector<size_t> hf_idx = makeIndex(hf_sample,d);

	XERUS_LOG(info, "Phi at HF sample "  << phi[hf_idx]);

	ContractionTree tree(phi,hf_sample);
	XERUS_LOG(info, "Phi at HF sample "  << tree.getValue());

	std::vector<size_t> sample = hf_sample;
	auto start = std::chrono::steady_clock::now();
	ContractionTree tree1h(phi,sample);
	for (size_t i = 0; i< test_number; ++i){
		sample = TrialSample(sample,d);
		auto tree = tree1h.updatedTree(sample);
		tree1h = new ContractionTree(phi,sample,tree);
	}
	auto end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for one hop  tree based evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");

	sample = hf_sample;
	start = std::chrono::steady_clock::now();
	for (size_t i = 0; i< test_number; ++i){
		sample = TrialSample(sample,d);
		phi[makeIndex(sample,d)];
	}
	end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for one hop linear evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");

	sample = hf_sample;
	start = std::chrono::steady_clock::now();
	ContractionTree tree2h(phi,sample);
	for (size_t i = 0; i< test_number; ++i){
		sample = TrialSample2(sample,d);
		auto tree = tree2h.updatedTree(sample);
		tree2h = new ContractionTree(phi,sample,tree);
	}
	end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for two hop  tree based evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");

	sample = hf_sample;
	start = std::chrono::steady_clock::now();
	for (size_t i = 0; i< test_number; ++i){
		sample = TrialSample2(sample,d);
		phi[makeIndex(sample,d)];
	}
	end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for two hop linear evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");


}
