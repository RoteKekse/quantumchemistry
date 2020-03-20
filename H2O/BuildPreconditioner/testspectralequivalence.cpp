#include <xerus.h>
#include "ALSres.cpp"

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
	size_t max_rank = 30;
	size_t max_rank_res = 30;
	Index ii,jj,kk,ll,mm;

	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading inverse of Fock Operator");
	TTOperator id,F,Finv;
	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);
	name = "../data/fock_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,F);
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info, "Loading shifted and preconditioned Hamiltonian");
	xerus::TTOperator Hs;
  name = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Hs);




	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi = TTTensor::random(std::vector<size_t>(2*nob, 2), {max_rank});
	phi /= phi.frob_norm();
	XERUS_LOG(info, "Norm phi start = " << phi.frob_norm());

	XERUS_LOG(info,"--- starting gradient descent ---");
	value_t rHx,rHr,rFx,rFr,xFx,xHx;
	double alpha;
	TTTensor res = TTTensor::random(std::vector<size_t>(2*nob, 2), {max_rank_res});
	TTTensor res_rounded = res;
	for (size_t iter = 0; iter < max_iter; ++iter){

		//update phi
		XERUS_LOG(info, "------ Iteration = " << iter);
		XERUS_LOG(info,"Gradient step");
		phi /= phi.frob_norm();
		xFx = contract_TT(F,phi,phi);
		xHx = contract_TT(Hs,phi,phi);
		XERUS_LOG(info, "lambda after rounding = " <<std::setprecision(8)<< xHx/xFx);
		getRes(Hs,phi,F,xHx/xFx,res);

		rHx = contract_TT(Hs,res,phi);
		rHr = contract_TT(Hs,res,res);
		rFx = contract_TT(F,res,phi);
		rFr = contract_TT(F,res,res);
		alpha = get_stepsize(xHx,rHr,rHx,xFx,rFr,rFx);

		phi = phi - alpha* res;
		phi.round(max_rank); //round phi

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
	XERUS_LOG(info,"alpha1 = " << alpha1);
	XERUS_LOG(info,"alpha2 = " << alpha2);
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

