#include <xerus.h>
#include "ALSres.cpp"
#include "solver.cpp"

using namespace xerus;
using xerus::misc::operator<<;
#define build_operator 0

void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);





int main(){
	size_t nob = 25;
	size_t num_elec = 10;
	double shift = 86.0;
	size_t max_iter = 1000;
	size_t max_rank = 10;
	Index ii,jj,kk,ll,mm;
	TTOperator Misr,Mi,id, Mi2,Mi3, Msr, H, Hs, Hsi;

	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading Inverse Square Root of Fock Operator");
	std::string name = "../data/Fock_inv_sqr_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Misr);

	XERUS_LOG(info, "Loading Square Root of Fock Operator");
	name = "../data/Fock_sqr_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Msr);

	XERUS_LOG(info, "Loading inverses of Fock Operator");
	name = "../data/Fock_inv_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Mi);
	name = "../data/DiagHamiltonian_inv_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Mi2);

	Mi3(ii/2,jj/2) = Misr(ii/2,kk/2) *  Misr(kk/2,jj/2);
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info, "--- Building Operator ---");
#if build_operator
	XERUS_LOG(info, "Loading Hamiltonian Operator");
	name = "../data/hamiltonian_H2O_" + std::to_string(2*nob) +"_full_2.ttoperator";
	std::ifstream read1(name.c_str());
	misc::stream_reader(read1,H,xerus::misc::FileFormat::BINARY);
	read1.close();

	XERUS_LOG(info, "Shifting Hamiltonian by " << shift);
	Hs = H + shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
	//H.round(0.0);

	std::string name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	write_to_disc(name2,Hs);

	XERUS_LOG(info, "Preconditioning Hamiltionian with inverse Square Root of Fock Operator");
	Hsi(ii/2,jj/2) = Misr(ii/2,kk/2) * Hs(kk/2,ll/2) * Misr(ll/2,jj/2);
	//H.round(0.0);
  name2 = "../data/hamiltonian_h2o_shifted_preconditioned" + std::to_string(2*nob) +"_full.ttoperator";
	write_to_disc(name2,Hsi);
	XERUS_LOG(info,"ranks of Hsi: " << Hsi.ranks());


#else
	XERUS_LOG(info, "Loading shifted and preconditioned Hamiltonian");
  std::string name2 = "../data/hamiltonian_h2o_shifted_preconditioned" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hsi);

  name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hs);
	XERUS_LOG(info,"ranks of Hsi: " << Hsi.ranks());
#endif

	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi(std::vector<size_t>(2*nob,2));
	xerus::TTTensor phi2(std::vector<size_t>(2*nob,2));


	phi = xerus::TTTensor::random(std::vector<size_t>(2*nob,2),std::vector<size_t>(2*nob-1,4));
	//XERUS_LOG(info, "Multiplying Start Vector by square root of Fock Operator");
	//phi(ii&0) = Msr(ii/2,jj/2) * phi(jj&0);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi2);
	XERUS_LOG(info,phi.ranks());

	XERUS_LOG(info,"--- starting gradient descent ---");
	Tensor tmp,rHx,rHr,rx;
	tmp()= phi(ii&0) * Hs(ii/2,jj/2) * phi(jj&0); //get ritz value

	XERUS_LOG(info,"Jacobi Davidson Intitialization ");
	solver sol(Hs,Misr,phi,-76.3 + shift - 8.80146457125193);
	sol.rounderror=1e-6;
	sol.maxranks=std::vector<size_t>(2*nob - 1,max_rank);
	sol.switchlambda=-76.0 + shift - 8.80146457125193;
	sol.CG_maxit=20;
	sol.CG_error=1e-4;

	solver sol2(Hs,id,phi2,-76.3 + shift - 8.80146457125193);
	sol2.rounderror=1e-6;
	sol2.maxranks=std::vector<size_t>(2*nob - 1,max_rank);
	sol2.switchlambda=-76.0 + shift - 8.80146457125193;
	sol2.CG_maxit=20;
	sol2.CG_error=1e-4;
	for (size_t iter = 0; iter < max_iter; ++iter){
		XERUS_LOG(info,"Step PC = "<< iter);
		sol.simple();
		XERUS_LOG(info, "lambda = " << sol.lambda  - shift + 8.80146457125193);
		XERUS_LOG(info,"Step No PC = "<< iter);
		sol2.simple();
		XERUS_LOG(info, "lambda = " << sol2.lambda  - shift + 8.80146457125193);

	}
	return 0;
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



