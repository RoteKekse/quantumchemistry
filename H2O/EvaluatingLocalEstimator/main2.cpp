#include <xerus.h>
//#include "hsclass_backup.cpp"
#include "hsclass3.cpp"
#include "metropolis.cpp"
#include <ctime>
#include <ratio>
#include <chrono>
#define test 0

using namespace xerus;
using xerus::misc::operator<<;
using namespace std::chrono;


int main(){
	XERUS_LOG(info, "---- Start building operator left to right! ----");
	XERUS_LOG(info, "Loading Trial Wave Function");
	xerus::TTTensor psi;
	std::string name = "../data/eigenvector_H2O_50_-85.191617_3.tttensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,psi,xerus::misc::FileFormat::BINARY);
	read.close();
	XERUS_LOG(info, "Loading Trial Wave Function -- Finished");
	XERUS_LOG(info,"Psi ranks = " << psi.ranks());



	size_t d = 50;
	size_t p = 10;
	std::string path_T = "../data/T_H2O_"+std::to_string(d)+".tensor";
	std::string path_V= "../data/V_H2O_"+std::to_string(d)+".tensor";
	value_t nuc = 8.80146457125193;

	XERUS_LOG(info, "Loading Hamiltonian Operator");
	xerus::TTOperator H;
	name = "../data/hamiltonian_H2O_" + std::to_string(d) +"_full_3.ttoperator";
	std::ifstream read1(name.c_str());
	misc::stream_reader(read1,H,xerus::misc::FileFormat::BINARY);
	read1.close();
	XERUS_LOG(info, "Loading Hamiltonian Operator -- Finished");


	Index i1,i2,j1,j2;
	Tensor testH, testE;
	TTTensor testS;
	value_t norm = 0;

	PsiHScontract builder(d,p,path_T,path_V,nuc);

  //std::vector<size_t> sample = builder.createRandomSample();
  std::vector<size_t> sample = {0,1,2,3,4,5,6,7,8,9};
  XERUS_LOG(info,"start contract");
	value_t res = builder.contract(psi,sample);
  XERUS_LOG(info,"end contract");

	Metropolis sample_gen(psi,sample);
	for (size_t i; i < 20; ++i)
		sample_gen.getNextSample();
	//TTOperator e1 = builder.build_brute(d);

	auto unit = builder.makeUnitVector(sample);
	testH() = unit(i1&0)*H(i1/2,i2/2)*psi(i2&0);
	testE() = unit(i1&0)*psi(i1&0);
	XERUS_LOG(info,"res from sum = " << res );
	XERUS_LOG(info,"res from op  = " << testH[0] / testE[0]);
//	XERUS_LOG(info,"res from op  = " << testE[0]);
//	XERUS_LOG(info,"res from op  = " << psi[{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}]);

	return 0;
}


