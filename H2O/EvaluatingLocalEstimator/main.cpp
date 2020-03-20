#include <xerus.h>
//#include "hsclass_backup.cpp"
#include "hsclass2.cpp"
#include "hsclass3.cpp"
#include <ctime>
#include <ratio>
#include <chrono>
#define test 0

using namespace xerus;
using xerus::misc::operator<<;
using namespace std::chrono;


int main(){
	XERUS_LOG(info, "---- Start building operator left to right! ----");

	size_t d = 116;
	size_t p = 10;
	std::string path_T = "../data/T_H2O_116.tensor";
	std::string path_V= "../data/V_H2O_116.tensor";
	value_t nuc = 8.80146457125193;

	Index i1,i2,j1,j2;
	Tensor testH, testE;
	TTTensor testS;
	value_t norm = 0;



#if test
	XERUS_LOG(test,"---- Testing");

	size_t test_dim = 8;
	//while (norm < 10e-10){
		HScontract tester(test_dim,4,path_T,path_V,nuc);
		//auto sample2 = tester.createRandomSample();
		std::vector<size_t> sample2 = { 1, 4, 5, 6 };//,1,2,3,4,5,6,7,8,9};
		XERUS_LOG(info,"sample = "<< sample2);
		auto unit_test = tester.makeUnitVector(sample2);
		auto res_test = tester.contract(sample2);

		auto testbuild = tester.build_brute(test_dim);
		testS(i1&0) = testbuild(i1/2,j1/2) * unit_test(j1&0);
		testH() = unit_test(i1&0)*testS(i1&0);
		norm = (testS-res_test).frob_norm();

		XERUS_LOG(info,norm);

		XERUS_LOG(info,"Norm testH " << testH);


		testE() = unit_test(j1&0) * res_test(j1&0);
		XERUS_LOG(info,"Norm testE " << testE);
		testS.round(0.0);
		res_test.round(0.0);
		XERUS_LOG(info,"ranks " << testS.ranks());
		XERUS_LOG(info,"ranks " << res_test.ranks());
	//}
#else
//		XERUS_LOG(info, "Loading Hamiltonian Operator");
//		xerus::TTOperator H;
//		std::string name = "../data/hamiltonian_H2O_" + std::to_string(d) +"_full_3.ttoperator";
//		std::ifstream read(name.c_str());
//		misc::stream_reader(read,H,xerus::misc::FileFormat::BINARY);
//		read.close();
//		XERUS_LOG(info, "Loading Hamiltonian Operator -- Finished");

		std::chrono::duration<double> runtime = std::chrono::duration<double>::zero();
		size_t idx = 0;
		HScontract builder(d,p,path_T,path_V,nuc);
	while (norm < 10e-9){
		++idx;
		std::vector<size_t> sample;
		XERUS_LOG(test,"Testing contract");
		sample = builder.createRandomSample();
		XERUS_LOG(info,"Iteration = " << idx);
		if (idx -1 == 0 ) sample = { 0, 2, 5,11, 20, 25, 28, 31, 44, 45 };
		auto unit = builder.makeUnitVector(sample);
    auto start = std::chrono::high_resolution_clock::now();
    auto res = builder.contract(sample);
    auto end = std::chrono::high_resolution_clock::now();
    XERUS_LOG(info,"sample = "<< sample);
    std::chrono::duration<double> diff = end-start;
    runtime += diff;
		XERUS_LOG(info,"diff: "<< diff.count()  << " sec");
		XERUS_LOG(info,"average time: "<< runtime.count() / idx  << " sec");

//		testS(i1&0) = H(i1/2,j1/2) * unit(j1&0);
//		testH() = unit(i1&0)*testS(i1&0);
//
//
//		testE() = unit(j1&0) * res(j1&0);
//
//		norm = (testS-res).frob_norm();
//		XERUS_LOG(info,"Norm err " << norm);
	}
#endif
	return 0;
}


