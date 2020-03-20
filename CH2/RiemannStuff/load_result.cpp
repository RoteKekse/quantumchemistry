#include <xerus.h>
#include "ALSres.cpp"


using namespace xerus;
using xerus::misc::operator<<;

int main() {

	xerus::TTTensor phi;

	std::string name = "../CH2res6.901e-13.tttensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,phi,xerus::misc::FileFormat::BINARY);
	read.close();

	xerus::TTOperator op;
	std::string name2 = "../hamiltonian_CH2_26_full.ttoperator";
	std::ifstream read2(name2.c_str());
	misc::stream_reader(read2,op,xerus::misc::FileFormat::BINARY);
	read.close();

	Index ii,jj,kk,ll,mm,nn,oo,i1,i2,i3,i4;
	Tensor ritz,ritz2;
	auto ones = Tensor::ones({1,1,1});
	auto ones2 = Tensor::ones({1,1,1,1});
	ritz = ones;
	for (size_t i = 0; i<26; ++i){
		XERUS_LOG(info, i);
		ritz(i1,i2,i3) = ritz(ii,jj,kk) * phi.get_component(i)(ii,mm,i1)*op.get_component(i)(jj,mm,nn,i2)*phi.get_component(i)(kk,nn,i3);
	}
	ritz() = ones(ii,jj,kk)*ritz(ii,jj,kk);

	auto ranks = std::vector<size_t>(26-1,600);
	TTTensor res = TTTensor::random(phi.dimensions,ranks);
	getRes(op, phi,ritz[0], res);


	XERUS_LOG(info, "phi ranks =" << phi.ranks());
	XERUS_LOG(info, "op ranks =" << op.ranks());
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );
	XERUS_LOG(info, "Ritz Value " << std::abs( ritz[0] -28.1930439210	+38.979392539208));
	XERUS_LOG(info, "Res = " << res.frob_norm());



	return 0;
}
