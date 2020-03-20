#include <xerus.h>

#include "../GradientMethods/tangentialOperation.cpp"
#include "../GradientMethods/basic.cpp"
#include "../loading_tensors.cpp"

using namespace xerus;
using xerus::misc::operator<<;

TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);
TTOperator particleNumberOperator(size_t d);
std::vector<size_t> makeRandomSample(size_t p,size_t d);

int main(){
	srand(time(NULL));

	Index i1,i2,i3,j1,j2,j3;
	size_t d = 48, num_elec = 8;
	std::vector<size_t> sample = makeRandomSample(num_elec,d);
	std::vector<size_t> sample1 = makeRandomSample(num_elec,d);
	std::vector<size_t> sample2 = makeRandomSample(num_elec,d);

	XERUS_LOG(info, "Sample  ek = " << sample);
	XERUS_LOG(info, "Sample1 ek = " << sample1);
	XERUS_LOG(info, "Sample2 ek = " << sample2);

	XERUS_LOG(info, "--- Loading Test Vector ---");
	TTTensor phi; 	TTOperator Hs;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	read_from_disc("../data/hamiltonian_H2O_48_full_benchmark.ttoperator",Hs);

	auto ek = makeUnitVector(sample,d) + makeUnitVector(sample1,d);


	TTOperator id = TTOperator::identity(std::vector<size_t>(2*d,2));
	auto ek2 = makeUnitVector(sample2,d);
	TangentialOperation top(phi);
	auto tang = top.localProduct(ek,id);
	XERUS_LOG(info, frob_norm(tang));

	auto tangTT = top.builtTTTensor(tang);
	auto PN = particleNumberOperator(d);

	XERUS_LOG(info, "--- Test PN OP ---");
  Tensor test; TTTensor test2;
//	test2(i1&0) = Hs(i1/2,j1/2)*ek(j1&0);
//	test() = test2(i1&0) *PN(i1/2,j1/2)*test2(j1&0);
//	XERUS_LOG(info,"PN of Hs*ek    " << test[0]/std::pow(test2.frob_norm(),2));

	test() = ek(i1&0) *PN(i1/2,j1/2)*ek(j1&0);
	XERUS_LOG(info,"PN of ek     " << test[0]/std::pow(ek.frob_norm(),2));


	test() = phi(i1&0) *PN(i1/2,j1/2)*phi(j1&0);
	XERUS_LOG(info,"PN of phi      " << test[0]/std::pow(phi.frob_norm(),2));

	test() = tangTT(i1&0) *PN(i1/2,j1/2)*tangTT(j1&0);
	XERUS_LOG(info,"PN of tang ek " <<test[0]/std::pow(tangTT.frob_norm(),2));
	return 0;
}


TTTensor makeUnitVector(std::vector<size_t> sample, size_t d){
	std::vector<size_t> index(d, 0);
	for (size_t i : sample)
		if (i < d)
			index[i] = 1;
	auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
	return unit;
}

TTOperator particleNumberOperator(size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n,{0,0,0,1});
	op.set_component(0,tmp);
	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
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
