#include <xerus.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <stdlib.h>

#include "../loading_tensors.cpp"
#include "probabilityfunctions.cpp"
#include "unitvectorprojection.cpp"

std::vector<size_t> makeRandomSample(size_t p,size_t d);


int main(){
	size_t d = 48, p = 8, position = 5, iterations = 1e7;
	std::string path_T = "../data/T_H2O_48_bench.tensor";
	std::string path_V= "../data/V_H2O_48_bench.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,5,22,23,31 },sample1 = { 0, 1,2,3,22,23,30,31 }, sample2;

	XERUS_LOG(info, "--- Loading Start Vector ---");
	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_5_-23.700175_benchmark.tttensor",phi);



	PsiProbabilityFunction PsiPF(phi);
  unitVectorProjection uvP(phi,position);


	XERUS_LOG(info,"Run test");
	for (size_t i = 0; i < iterations; ++i) {

		std::vector<size_t> sample = makeRandomSample(p,d);
		auto loc_grad = uvP.localProduct(sample);
		auto idx = PsiPF.makeIndex(sample);

		if (loc_grad.frob_norm() > 1e-4 || phi[idx] > 1e-4){
			XERUS_LOG(info,sample << ":");
			XERUS_LOG(info,"(Pek,xek) = (" << loc_grad.frob_norm() << ", " << phi[idx]<< ")");
		}
		if (i % (iterations/100) == 0)
			XERUS_LOG(info, i);

	}



	return 0;
}



std::vector<size_t> makeRandomSample(size_t p,size_t d){
	 std::vector<size_t> sample;
		while(sample.size() < p){
			auto r = rand() % (d);
			auto it = std::find (sample.begin(), sample.end(), r);
			if (it == sample.end()){
				sample.emplace_back(r);
			}
		}
		sort(sample.begin(), sample.end());

		return sample;
}
