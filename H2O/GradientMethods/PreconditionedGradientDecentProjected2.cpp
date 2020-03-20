#include <xerus.h>
#include "tangentialOperation.cpp"
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
class localProblem;




int main(){
	size_t nob = 25;
	size_t num_elec = 10;
	double shift = 86.0;
	size_t max_iter = 1000;
	size_t max_rank = 30;
	Index ii,jj,kk,ll,mm;
	value_t scale = 0.3;

	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading inverse of Fock Operator");
	TTOperator id,Finv;
	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);

	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
  std::string name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hs);


	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi;


	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi);

  XERUS_LOG(info,"--- starting gradient descent ---");
	value_t rHx,rHr,rx,rr,xx,xHx;
	double alpha;
  time_t begin_time,global_time = time (NULL);
  std::ofstream outfile;
	std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/PreconProjected2_rank_" + std::to_string(max_rank) +".csv";
	outfile.open(out_name);
	outfile.close();

	TTTensor res;
	std::vector<std::pair<time_t,value_t>> result;
	std::vector<Tensor> res_tangential;
	TangentialOperation Top(phi);
	for (size_t iter = 0; iter < max_iter; ++iter){
		//update phi
		XERUS_LOG(info, "------ Iteration = " << iter);
		XERUS_LOG(info,"Projected Gradient step with PC (non symmetric)");
		xx = phi.frob_norm();
		phi /= xx; //normalize
		xHx = contract_TT(Hs,phi,phi);
		result.emplace_back(std::pair<time_t,value_t>(time (NULL)-global_time,xHx  - shift + 8.80146457125193));

		//Write to file
		outfile.open(out_name,std::ios::app);
	  outfile << time (NULL)-global_time << "," <<std::setprecision(12) << xHx  - shift + 8.80146457125193 << std::endl;
	  outfile.close();

		XERUS_LOG(info, "lambda after rounding = " <<std::setprecision(8)<< xHx  - shift + 8.80146457125193);
		begin_time = time (NULL);
		res_tangential.clear();
		res_tangential = Top.localProduct(Hs,id,xHx);
		XERUS_LOG(info,"---Time for get res_tangential: " << time (NULL)-begin_time<<" sekunden");
		begin_time = time (NULL);
		res = Top.builtTTTensor(res_tangential);
		XERUS_LOG(info,res.frob_norm());
		XERUS_LOG(info,"---Time for get res: " << time (NULL)-begin_time<<" sekunden");

		begin_time = time (NULL);
		res_tangential.clear();
		res_tangential = Top.localProduct(res,Finv);
		XERUS_LOG(info,"---Time for get res_tangential2: " << time (NULL)-begin_time<<" sekunden");
		begin_time = time (NULL);
		res = Top.builtTTTensor(res_tangential);
		XERUS_LOG(info,res.frob_norm());
		XERUS_LOG(info,"---Time for get res2: " << time (NULL)-begin_time<<" sekunden");

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
		Top.update(phi);
		XERUS_LOG(info,std::setprecision(16) << result);
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




