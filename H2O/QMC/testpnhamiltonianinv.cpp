#include <xerus.h>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>



using namespace xerus;
using xerus::misc::operator<<;


void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
void read_from_disc(std::string name, Tensor &x);
TTOperator tang_proj(TTTensor &tang_root, size_t pos);
void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
TTOperator particleNumberOperator(size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);
std::vector<size_t> makeRandomSample(size_t p,size_t d);

/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for Particle Number Invariance of the Hamiltonian ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");
	Index ii,jj,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2;
	//Set Parameters
	size_t d = 48;
	size_t part = 48 ;
	size_t p = 8;
	size_t nob = part / 2;
	double eps = 10e-10;
	size_t start_rank = 1;
	size_t max_rank = 30;
	size_t number_of_samples = 10000;
	size_t pos = 3;


	xerus::TTTensor phi(std::vector<size_t>(part,2)),phitmp;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	phi.move_core(pos);
	XERUS_LOG(info,"corePos " << phi.corePosition);
	XERUS_LOG(info,"\n" <<phi.ranks());
	XERUS_LOG(info, phi.frob_norm());
	//Load Hamiltonian
	xerus::TTOperator op;
	phitmp = phi;
	read_from_disc("../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_benchmark.ttoperator",op);

	project(phi,p,d);
	project(phi,p,d);
	project(phi,p,d);

	XERUS_LOG(info,"\n" <<phi.ranks());

	phi.round(8);
	phi.round(1e-8);
	XERUS_LOG(info,"\n" <<phi.ranks());
	phi.canonicalized = false;
	phi.move_core(pos);
	XERUS_LOG(info,"rounding ad projection error " <<(phi-phitmp).frob_norm());
	//phi /=phi.frob_norm();
	//Calculate particle number
	auto P = particleNumberOperator(d);
	auto Pphi = tang_proj(phi,pos);
	TTOperator PPphi,PphiP;
	PPphi(i1/2,j1/2) = P(i1/2,k1/2) *Pphi(k1/2,j1/2);
	PphiP(i1/2,j1/2) = Pphi(i1/2,k1/2) *P(k1/2,j1/2);
	PPphi.move_core(0);
	PphiP.move_core(0);
	XERUS_LOG(info,"Commutator = " << (PPphi- PphiP).frob_norm());

	auto t0 = Pphi.get_component(0);
	auto t1 = Pphi.get_component(1);
	auto t2 = Pphi.get_component(2);

	t0(i1,i2,i3,i4,j1,j2,j3,j4) = t0(i1,i2,j2,k1)*t1(k1,i3,j3,k2)*t2(k2,i4,j4,j1);
	t0.reinterpret_dimensions({8,8});
	XERUS_LOG(info,"\n" <<t0);

	Tensor pn,pn2;
	phitmp /= phitmp.frob_norm();
	pn() = P(ii/2,jj/2)*phitmp(ii&0)*phitmp(jj&0);
	XERUS_LOG(info," PN of phi before= " << std::setprecision(15)<< pn[0]);

	phi /= phi.frob_norm();
	pn() = P(ii/2,jj/2)*phi(ii&0)*phi(jj&0);
	XERUS_LOG(info," PN of phi after = " << std::setprecision(15)<< pn[0]);



//	for (size_t i = 0; i < number_of_samples; ++i){
//		auto s1 = makeRandomSample(p,d);
//		auto s2 = makeRandomSample(p,d);
//		auto ek = makeUnitVector(s1,d);
//		auto el = makeUnitVector(s2,d);
//		Tensor tt;
//		tt() = ek(ii&0)*Pphi(ii/2,jj/2) *el(jj&0);
//		if (std::abs(tt[0]) > 1e-16)
//			XERUS_LOG(info,"val = " << tt[0] << " ek = " << s1 << " el = " << s2 );
//	}







	return 0;
}

TTOperator tang_proj(TTTensor &tang_root, size_t pos){
	size_t d = tang_root.order();
	tang_root.move_core(pos);
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	TTOperator Py = TTOperator(std::vector<size_t>(2*d,2)),tto1,tto2;
	auto id2 = Tensor::identity({2,2});
	id2.reinterpret_dimensions({1,2,2,1});
	Py.set_component(pos,id2);
	for (size_t i = 0; i < pos; ++i){
		auto ti = tang_root.get_component(i);
		if (i == pos-1){
			ti(i1,j1,i2,j2) = ti(i1,i2,k1) * ti(j1,j2,k1);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],1});
		}
		else{
			ti(i1,j1,i2,j2,i3,j3) = ti(i1,i2,i3) * ti(j1,j2,j3);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],dim[4]*dim[5]});
		}
		Py.set_component(i,ti);
	}
	for (size_t i = pos+1; i < d; ++i){
		auto ti = tang_root.get_component(i);
		if (i == pos+1){
			ti(i2,j2,i3,j3) = ti(k1,i2,i3) * ti(k1,j2,j3);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({1,dim[0],dim[1],dim[2]*dim[3]});
		}
		else{
			ti(i1,j1,i2,j2,i3,j3) = ti(i1,i2,i3) * ti(j1,j2,j3);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],dim[4]*dim[5]});
		}
		Py.set_component(i,ti);
	}
	return Py;
}


void write_to_disc(std::string name, TTOperator &op){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
	write.close();
}

void read_from_disc(std::string name, TTOperator &op){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();

}

void read_from_disc(std::string name, TTTensor &x){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,x,xerus::misc::FileFormat::BINARY);
	read.close();

}

void read_from_disc(std::string name, Tensor &x){
	std::ifstream read(name.c_str() );
	xerus::misc::stream_reader(read,x,xerus::misc::FileFormat::BINARY);
	read.close();

}

void project(TTTensor &phi, size_t p, size_t d){
	Index i1,i2,j1,j2,k1,k2;
	for (size_t k = 0; k <= d; ++k){

		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t) p - (value_t) k;
			phi /=  f;
			//phi.round(1e-12);
			phi /= phi.frob_norm();

		}
	}
	phi /= phi.frob_norm();
}



TTOperator particleNumberOperator(size_t k, size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;

	value_t kk = (value_t) k;
	auto kkk = kk/(value_t) d * id;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n-kkk,{0,0,0,1});
	op.set_component(0,tmp);

	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n-kkk,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n-kkk,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
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



TTTensor makeUnitVector(std::vector<size_t> sample, size_t d){
	std::vector<size_t> index(d, 0);
	for (size_t i : sample)
		if (i < d)
			index[i] = 1;
	auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
	return unit;
}
