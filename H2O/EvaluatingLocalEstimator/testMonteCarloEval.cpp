#include <xerus.h>

#include "hsclass3.cpp"
#include "metropolis.cpp"
#include <unordered_map>
using namespace xerus;
using xerus::misc::operator<<;

template <typename Container> // we can make this generic for any container [1]
struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

std::vector<size_t> makeIndex(std::vector<size_t> sample, size_t d){
	std::vector<size_t> index(d, 0);
	for (size_t i : sample)
		if (i < d)
			index[i] = 1;
	return index;
}

int main(){
	XERUS_LOG(info, "--- Test Monte Carlo Evaluation for given psi and H");
	size_t d = 48;
	size_t p = 10;
	std::string path_T = "../data/T_H2O_"+std::to_string(d)+"_bench.tensor";
	std::string path_V= "../data/V_H2O_"+std::to_string(d)+"_bench.tensor";
	value_t nuc = 8.80146457125193;
	std::vector<size_t> sample ={ 0, 1, 2, 3,22,23,30,31 };
	std::vector<size_t> hf = sample;
	PsiHScontract builder(d,p,path_T,path_V,nuc);

	XERUS_LOG(info, "Loading Trial Wave Function");
	xerus::TTTensor psi;
	std::string name = "../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,psi,xerus::misc::FileFormat::BINARY);
	read.close();
	XERUS_LOG(info, "Loading Trial Wave Function -- Finished");

	XERUS_LOG(info, "Loading Hamiltonian Operator");
	xerus::TTOperator H;
	name = "../data/hamiltonian_H2O_48_full_benchmark.ttoperator";
	std::ifstream read1(name.c_str());
	misc::stream_reader(read1,H,xerus::misc::FileFormat::BINARY);
	read1.close();
	XERUS_LOG(info, "Loading Hamiltonian Operator -- Finished");

	Index i1,i2,j1,j2;
	Tensor testH, testE,testS;
	TTTensor unit = builder.makeUnitVector(hf,d);
	testH() = psi(i1&0)*H(i1/2,i2/2)*psi(i2&0);
	testE() = psi(i1&0)*H(i1/2,i2/2)*unit(i2&0);
	testS() = psi(i1&0)*unit(i1&0);
	XERUS_LOG(info,"True Energy  = " << testH[0]);
	XERUS_LOG(info,"HF Energy  = " << testE[0]*testS[0]);
	XERUS_LOG(info,"Norm psi  = " << psi.frob_norm());


	size_t N = 0;
	value_t Ntilde = 0;

  value_t energy = 0;
  value_t loc_energy = 0;
	Metropolis sample_gen(psi,sample);
	std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> umap;
	value_t psi_max = 0;
	//value_t hf_energy = tes;
	umap[hf] = testE[0]/testS[0];
	for (size_t i = 0; i < 10000; ++i)
		sample = sample_gen.getNextSample();
  XERUS_LOG(info,"got next sample");
	while(N < 10000000){
		++N;
		auto itr = umap.find(sample);
		if (itr == umap.end()){
			builder.reset();
			loc_energy = builder.contract(psi,sample);

			umap[sample] = loc_energy;
			energy += loc_energy;
			//XERUS_LOG(info,"Calc val, sample  " << sample  << " eng = " << loc_energy);
		}
		else{
			//XERUS_LOG(info,"Found val, sample " << sample  << " eng = " << itr->second);
			energy += itr->second;
		}
		if (N%10 == 0) {
			Ntilde = (value_t) N;
			value_t en_tmp = energy/Ntilde ;
			XERUS_LOG(info, "Energy after " << N << " steps = " << en_tmp  << " err = " << std::abs(testH[0]-en_tmp));
		}
		sample = sample_gen.getNextSample();
		auto tmp_val = psi[makeIndex(sample,d)];
		if (tmp_val*tmp_val > psi_max){
			psi_max = tmp_val*tmp_val;
			XERUS_LOG(info,tmp_val);
			XERUS_LOG(info,psi_max);
			XERUS_LOG(info,sample);
		}

	}



	return 0;

}
