#include <xerus.h>
#include "ALSres.cpp"
#include "solver.cpp"

using namespace xerus;
using xerus::misc::operator<<;
#define build_operator 0

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
void project(std::vector<Tensor>& x,std::vector<Tensor>& y,const std::vector<Tensor>& Q);
value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y);





int main(){
	size_t nob = 25;
	size_t num_elec = 10;
	double shift = 86.0;
	size_t max_iter = 1000;
	size_t max_rank = 60;
	size_t max_rank_res = 150;
	Index ii,jj,kk,ll,mm;
	value_t scale = 0.3;

	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading Inverse Square Root of Fock Operator");
	TTOperator Misr;
	std::string name = "../data/Fock_inv_sqr_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Misr);
	XERUS_LOG(info,"first entry: " << Misr[	std::vector<size_t>(4*nob,0)]);

	XERUS_LOG(info, "Loading Square Root of Fock Operator");
	TTOperator Msr;
	name = "../data/Fock_sqr_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Msr);

	XERUS_LOG(info,"first entry: " << Msr[	std::vector<size_t>(4*nob,0)]);
	XERUS_LOG(info, "Loading inverse of Fock Operator");
	TTOperator Mi,id, Mi2,Mi3;
	name = "../data/Fock_inv_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Mi);

	name = "../data/DiagHamiltonian_inv_H2O_" + std::to_string(2*nob) +".ttoperator";
	read_from_disc(name,Mi2);

	name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Mi3);

	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info, "--- Building Operator ---");

#if build_operator
	XERUS_LOG(info, "Loading Hamiltonian Operator");
	xerus::TTOperator H, Hs, Hsi;
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
	xerus::TTOperator Hs;

  std::string name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hs);

#endif

	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi,phi2,phi3,phi4,phi5;


	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi2);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi3);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi4);
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi5);

//	XERUS_LOG(info,"Jacobi Davidson Intitialization ");
//	solver sol(Hs,Misr,phi4,-76.3 + shift - 8.80146457125193);
//	sol.rounderror=1e-6;
//	sol.maxranks=std::vector<size_t>(2*nob - 1,max_rank);
//	sol.switchlambda=-76.0 + shift - 8.80146457125193;
//	sol.CG_maxit=20;
//	sol.CG_error=1e-4;
//
//	solver sol2(Hs,id,phi5,-76.3 + shift - 8.80146457125193);
//	sol2.rounderror=1e-6;
//	sol2.maxranks=std::vector<size_t>(2*nob - 1,max_rank);
//	sol2.switchlambda=-76.0 + shift - 8.80146457125193;
//	sol2.CG_maxit=20;
//	sol2.CG_error=1e-4;


	XERUS_LOG(info,"--- starting gradient descent ---");
	value_t rHx,rHr,rx,rr,xx,xHx;
	Tensor tmp;
	tmp()= phi(ii&0) * Hs(ii/2,jj/2) * phi(jj&0); //get ritz value
	double alpha;
  time_t begin_time,global_time = time (NULL);
	TTTensor res2 = TTTensor::random(std::vector<size_t>(2*nob, 2), {max_rank_res});
	TTTensor res = res2,res_rounded = res2;
	TTTensor res3 = res2;
	std::vector<value_t> result;

  std::ofstream outfile;
	std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/PreconRounded_rank_" + std::to_string(max_rank) +".csv";
	outfile.open(out_name);
	outfile.close();

	for (size_t iter = 0; iter < max_iter; ++iter){

		//update phi
		XERUS_LOG(info, "------ Iteration = " << iter);
//		XERUS_LOG(info,"Gradient step without PC");
//		begin_time = time (NULL);
//		xx =  phi2.frob_norm();
//		phi2 /= xx; //normalize
//		xHx = contract_TT(Hs,phi2,phi2);
//		XERUS_LOG(info, "lambda = " << xHx  - shift + 8.80146457125193);
//		getRes(Hs,phi2,id,xHx,res2);
//		XERUS_LOG(info, "res norm  = " << res2.frob_norm());
//		rHx = contract_TT(Hs,res2,phi2);
//		rHr = contract_TT(Hs,res2,res2);
//		rx = contract_TT(id,res2,phi2);
//		rr = contract_TT(id,res2,res2);
//		alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
//		phi2 = phi2 - alpha * res2;
//		phi2.round(max_rank); //round phi

		XERUS_LOG(info,"Gradient step with PC (non symmetric)");
		xx = phi.frob_norm();
		phi /= xx; //normalize
		xHx = contract_TT(Hs,phi,phi);
		result.emplace_back(xHx  - shift + 8.80146457125193);
		//Write to file
		outfile.open(out_name,std::ios::app);
		outfile << time (NULL)-global_time << "," <<std::setprecision(12) << xHx  - shift + 8.80146457125193 << std::endl;
		outfile.close();

		XERUS_LOG(info, "lambda after rounding = " <<std::setprecision(8)<< xHx  - shift + 8.80146457125193);
		begin_time = time (NULL);
		getRes(Hs,phi,id,xHx,res);
		XERUS_LOG(info,"---Time for get res: " << time (NULL)-begin_time<<" sekunden");
		begin_time = time (NULL);
		getRes(Mi3,res,id,0,res_rounded);
		XERUS_LOG(info,"---Time for preconditioning: " << time (NULL)-begin_time<<" sekunden");

		res = res_rounded;
		begin_time = time (NULL);
		rHx = contract_TT(Hs,res,phi);
		rHr = contract_TT(Hs,res,res);
		rx = contract_TT(id,res,phi);
		rr = contract_TT(id,res,res);
		alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
		XERUS_LOG(info,"---Time for alpha: " << time (NULL)-begin_time<<" sekunden");
		begin_time = time (NULL);
		XERUS_LOG(info,"---Time for step: " << time (NULL)-begin_time<<" sekunden");
		phi = phi - alpha* res;
//			phi = phi +  res;
		begin_time = time (NULL);
		phi.round(max_rank); //round phi
		XERUS_LOG(info,"---Time for round: " << time (NULL)-begin_time<<" sekunden");
		XERUS_LOG(info,std::setprecision(10) <<result);

//		XERUS_LOG(info,"Gradient step with PC (symmetric)");
//		TTTensor phiM, resM;
//		phiM(ii&0) = Misr(ii/2,jj/2) * phi3(jj&0);
//		xx = phiM.frob_norm();
//		phi3 /= xx;//normalize
//		phiM /= xx;//normalize
//		xHx = contract_TT(Hs,phiM,phiM);
//		XERUS_LOG(info, "lambda = " << xHx  - shift + 8.80146457125193);
//		getRes(Hs,phi3,Misr,xHx,res3);
//		XERUS_LOG(info, "res norm  = " << res3.frob_norm());
//		resM(ii&0) = Misr(ii/2,jj/2) * res3(jj&0);
//		rHx = contract_TT(Hs,resM,phiM);
//		rHr = contract_TT(Hs,resM,resM);
//		rx = contract_TT(id,resM,phiM);
//		rr = contract_TT(id,resM,resM);
//		alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
//		phi3 = phi3 - alpha* res3;
//		phi3.round(max_rank); //round phi


//		XERUS_LOG(info,"Jacobi Davidson step");
//		begin_time = time (NULL);
//		sol.simple();
//		XERUS_LOG(info, "lambda = " << sol.lambda  - shift + 8.80146457125193);
//		XERUS_LOG(info,"Time for step: " << time (NULL)-begin_time<<" sekunden");
//
//		XERUS_LOG(info,"Jacobi Davidson step with PC (symmetric)");
//		begin_time = time (NULL);
//		sol2.simple();
//		XERUS_LOG(info, "lambda = " << sol2.lambda  - shift + 8.80146457125193);
//		XERUS_LOG(info,"Time for step: " << time (NULL)-begin_time<<" sekunden");

	}
	return 0;
}

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx){
	double a = rFr*rHx-rHr*rFx;
	double b = rHr*xFx-rFr*xHx;
	double c = rFx*xHx-rHx*xFx;

	double disc = b*b-4*a*c;
	double alpha1 = (-b + std::sqrt(disc))/(2*a);
	double alpha2 = (-b - std::sqrt(disc))/(2*a);
	return alpha1;
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

void project(std::vector<Tensor>& x, std::vector<Tensor>& y, const std::vector<Tensor>& Q){
    size_t d=x.size();
    for (size_t pos=0; pos<d;pos++){
        Index i;
        Tensor tmp;
        tmp()=x[pos](i&0)*Q[pos](i&0);
        x[pos]-=tmp[0]*y[pos];
    }
}

//
//double innerprod(const std::vector<Tensor>& y,const std::vector<Tensor>& z){
//    double result=0;
//    size_t d=y.size();
//    for (size_t pos=0; pos<d;pos++){
//        Index i;
//        Tensor tmp;
//        tmp()=y[pos](i&0)*z[pos](i&0);
//        result+=tmp[0];
//    }
//    return result;
//}

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

