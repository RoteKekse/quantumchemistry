#include <xerus.h>
#include <ctime>
#include "tangential.cpp"

std::vector<size_t> makeRandomSample(size_t p,size_t d);


int main(){
	srand(time(NULL));
	size_t d = 48, p = 8, iterations = 1e5, pos = 5;
	std::vector<size_t> sample = { 0, 3, 8, 13, 22, 23, 30, 31 };
	clock_t time_req;


	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_5_-23.700175_benchmark.tttensor",phi);


	ProjectorProbabilityFunction PPF(phi,pos, true);
	//create test set
	std::vector<std::vector<size_t>> samples;
	for (size_t i = 0; i < iterations;++i)
		samples.emplace_back(makeRandomSample(p,d));


	time_req = clock();
	for (auto s : samples){
		PPF.localProduct(s);
	}
	time_req = clock() - time_req;
	XERUS_LOG(info,"Method 1 took " << (value_t)time_req/CLOCKS_PER_SEC << " secs");

//	time_req = clock();
//	for (auto s : samples){
//		PPF.localProduct2(s);
//	}
//	time_req = clock() - time_req;
//	XERUS_LOG(info,"Method 2 took " << (value_t)time_req/CLOCKS_PER_SEC << " secs");
//
//	time_req = clock();
//	size_t count = 0;
//	for (auto s : samples){
//		if  ((PPF.localProduct2(s)-PPF.localProduct(s)).frob_norm() < 1e-40)
//			//XERUS_LOG(info,(PPF.localProduct2(s)-PPF.localProduct(s)).frob_norm());
//			count++;
//	}
//	time_req = clock() - time_req;
//	XERUS_LOG(info,"Error took " << (value_t)time_req/CLOCKS_PER_SEC << " secs" << " count " << count << " of " << samples.size());

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
