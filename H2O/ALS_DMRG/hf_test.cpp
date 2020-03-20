#include <xerus.h>




using namespace xerus;
using xerus::misc::operator<<;





/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");


	size_t d = 48; // 16 electron, 8 electron pairs
	size_t nob = 24;
	double eps = 10e-12;
	size_t start_rank = 2;
	size_t max_rank = start_rank;
	Index ii,jj;

	xerus::TTOperator op;
	std::string name = "../hamiltonian_H2O_" + std::to_string(d) +"_full_3.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
	read.close();

	xerus::TTTensor phi(std::vector<size_t>(d,2));
	phi.canonicalized = false;

	std::vector<size_t> v(d-10);
	std::iota(v.begin(), v.end(), 10);

	for (size_t i : {0,1,2,3,4,5,6,7,8,9}){
		phi.component(i)[{0,0,0}] = 0.0;
		phi.component(i)[{0,1,0}] = 1.0;
	}
	for (size_t i : v){
		phi.component(i)[{0,0,0}] = 1.0;
		phi.component(i)[{0,1,0}] = 0.0;
	}



	Tensor r,r2, phitest;
	r() = phi(ii&0)*phi(jj&0) * op(ii/2,jj/2);
	r2() = phi(ii&0)*phi(ii&0);
	XERUS_LOG(info, "r = " << r[0]);
	XERUS_LOG(info, "r = " << r[0]/r2[0] - 52.4190597253);
	XERUS_LOG(info, "phi = " << phi.frob_norm());
	XERUS_LOG(info, "phi = " << phi.ranks());



	return 0;
}
