#include <xerus.h>
#include "ALSres.cpp"
#include "Stochres.cpp"
#include "solver.cpp"

using namespace xerus;
using xerus::misc::operator<<;
#define build_operator 0

double get_stepsize(double lambda,double a2, double a3, double b2);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y);




int main(){
	size_t nob = 25;
	size_t num_elec = 10;
	double shift = 86.0;
	size_t max_iter = 1000;
	size_t max_rank = 30;
	size_t res_rank = 30;
	size_t precond = 1;
	double eps = 10e-5;

	Index ii,jj,kk,ll,mm;

	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading Inverse Square Root of Fock Operator");
	TTOperator Misr;
	std::string name = "../data/Fock_inv_sqr_H2O_" + std::to_string(2*nob) +"rank"+ std::to_string(precond)+ ".ttoperator";
	read_from_disc(name,Misr);

	TTOperator id;
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info, "--- Building Operator ---");

#if build_operator
	XERUS_LOG(info, "Loading Hamiltonian Operator");
	xerus::TTOperator H, Hs;
	name = "../data/hamiltonian_H2O_" + std::to_string(2*nob) +"_full_2.ttoperator";
	std::ifstream read1(name.c_str());
	misc::stream_reader(read1,H,xerus::misc::FileFormat::BINARY);
	read1.close();

	XERUS_LOG(info, "Shifting Hamiltonian by " << shift);
	Hs = H + shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
	//H.round(0.0);

	std::string name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	write_to_disc(name2,Hs);

#else
	XERUS_LOG(info, "Loading shifted Hamiltonian");
	xerus::TTOperator Hs;
	std::string  name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hs);
#endif

	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi,phi2,phi3,phi4,phi5;


	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi3);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi4);



	XERUS_LOG(info,"--- starting gradient descent ---");
	value_t tmp,rHx,rHr,rx;

	double alpha;
  time_t begin_time;
	TTTensor res3 = TTTensor::random(std::vector<size_t>(2*nob, 2), {res_rank});
	TTTensor res4 = res3;
	for (size_t iter = 0; iter < max_iter; ++iter){
		TTTensor phiM, resM;

		//update phi
		XERUS_LOG(info, "------ Iteration = " << iter);
		XERUS_LOG(info,"Gradient step with PC (symmetric)");
		begin_time = time (NULL);
		phiM(ii&0) = Misr(ii/2,jj/2) * phi3(jj&0);
		tmp = phiM.frob_norm();
		phi3 /= tmp;//normalize
		XERUS_LOG(info,"---Time after normalization: " << time (NULL)-begin_time<<" sekunden");
		phiM /= tmp;
		tmp = contract_TT(Hs,phiM,phiM);
		XERUS_LOG(info,"---Time after lambda calc: " << time (NULL)-begin_time<<" sekunden");
		XERUS_LOG(info, "---lambda = " << std::setprecision(12) << tmp  - shift + 8.80146457125193);

		getRes(Hs,phi3,Misr,tmp,res3);
		//getRes_Stoch(Hs,phi3,Misr,tmp,res,10,10);
		XERUS_LOG(info,"---Time after get res: " << time (NULL)-begin_time<<" sekunden");

		resM(ii&0) = Misr(ii/2,jj/2) * res3(jj&0);
		tmp = resM.frob_norm();
		res3 /= tmp;
		alpha = 0.01;
//		resM(ii&0) = Misr(ii/2,jj/2) * res(jj&0);
//		rHx = contract_TT(Hs,resM,phiM);
//		rHr = contract_TT(Hs,resM,resM);
//		rx = contract_TT(id,resM,phiM);
//		alpha = get_stepsize(tmp,rHx,rHr,rx);
		XERUS_LOG(info, "---Stepsize = " << alpha);
		XERUS_LOG(info,"---Time after get stepsize: " << time (NULL)-begin_time<<" sekunden");
		phi3 = phi3 - alpha* res3;
		XERUS_LOG(info,"---Time after update: " << time (NULL)-begin_time<<" sekunden");
		phi3.round(std::vector<size_t>(2*nob-1, max_rank), eps); //round phi
		XERUS_LOG(info,"---Time: " << time (NULL)-begin_time<<" sekunden");


		//update phi
		XERUS_LOG(info,"Gradient step with PC (non - symmetric)");
		begin_time = time (NULL);
		phi4 /= phi4.frob_norm();//normalize
		XERUS_LOG(info,"---Time after normalization: " << time (NULL)-begin_time<<" sekunden");
		tmp = contract_TT(Hs,phi4,phi4);
		XERUS_LOG(info,"---Time after lambda calc: " << time (NULL)-begin_time<<" sekunden");
		XERUS_LOG(info, "---lambda = " << std::setprecision(12) << tmp  - shift + 8.80146457125193);

		getRes(Hs,phi4,id,tmp,res4);
		//getRes_Stoch(Hs,phi3,Misr,tmp,res,10,10);
		XERUS_LOG(info,"---Time after get res: " << time (NULL)-begin_time<<" sekunden");

		res4 /= res4.frob_norm();
		alpha = 0.01;
		XERUS_LOG(info, "---Stepsize = " << alpha);
		XERUS_LOG(info,"---Time after get stepsize: " << time (NULL)-begin_time<<" sekunden");
		phi4 = phi4 - alpha* res4;
		XERUS_LOG(info,"---Time after update: " << time (NULL)-begin_time<<" sekunden");
		phi4.round(std::vector<size_t>(2*nob-1, max_rank), eps); //round phi
		XERUS_LOG(info,"---Time: " << time (NULL)-begin_time<<" sekunden");
	}

	return 0;
}

double get_stepsize(double lambda,double a2, double a3, double b2){
	double a = a2 - a3 * b2;
	double b = a3 -lambda*b2;
	double c = lambda*b2 -a2;

	double disc = b*b-4*a*c;
	double alpha = (-b + std::sqrt(disc))/(2*a);
	return alpha;
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



value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y){
	Tensor stack = Tensor::ones({1,1,1});
	size_t d = A.order()/2;
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	for (size_t i = 0; i < d; ++i){
		auto Ai = A.get_component(i);
		auto xi = x.get_component(i);
		auto yi = y.get_component(i);

		stack(i1,i2,i3) = stack(j1,j2,j3) * xi(j1,k1,i1) * Ai(j2,k1,k2,i2) * yi(j3,k2,i3);
	}
	return stack[0,0,0];
}



