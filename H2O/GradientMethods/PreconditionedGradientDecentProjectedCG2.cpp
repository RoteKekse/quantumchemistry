#include <xerus.h>
#include "tangentialOperation.cpp"
#include "ALSres.cpp"
#include "basic.cpp"

using namespace xerus;
using xerus::misc::operator<<;
#define build_operator 0

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);
void write_to_disc(std::string name, TTOperator &op);
void write_to_disc(std::string name, TTTensor &x);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
void project(std::vector<Tensor>& x,std::vector<Tensor>& y,const std::vector<Tensor>& Q);
value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y);
class localProblem;




int main(){
	size_t nob = 25;
	size_t num_elec = 10;
	double shift = 86.0;
	size_t max_iter = 50;
	size_t max_rank = 60;
	Index ii,jj,kk,ll,mm;
	value_t eps = 10e-6;
	value_t roh = 0.5, c1 = 1e-4;
	bool round = false;
	size_t rank_precon = 0;
	value_t alpha_start = 2;

	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading inverse of Fock Operator");
	TTOperator id,Finv;
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);
	Finv = (-1)*Finv;
	if (round){
		if (rank_precon == 0)
			Finv = id;
		else
			Finv.round(rank_precon);
	}


	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
  std::string name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hs);


	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi,phi_tmp;


	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",phi);

  XERUS_LOG(info,"--- starting gradient descent ---");
	value_t rHx,rHr,rx,rr,xx,xHx, xHx_tmp, xx_tmp, projection_time,stepsize_time,eigenvalue_time;
	double alpha,beta,residual;
  clock_t begin_time,global_time = clock();


	TTTensor res,res1,res2, res_last;
	std::vector<value_t> result;
	std::vector<Tensor> res_tangential, res_last_tangential;
	TangentialOperation Top(phi);
  std::ofstream outfile;
	std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/PreconProjectedCG2_rank_" + std::to_string(max_rank) +"_cpu_exact.csv";
	if (round)
		out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/PreconProjectedCG2_rank_" + std::to_string(max_rank) +"_rd_"+std::to_string(rank_precon)+"_cpu_exact.csv";
	outfile.open(out_name);
	outfile.close();

	xx = phi.frob_norm();
	phi /= xx; //normalize
	xHx = contract_TT(Hs,phi,phi);
	result.emplace_back(xHx  - shift + 8.80146457125193);
	XERUS_LOG(info, "lambda after rounding = " <<std::setprecision(8)<< xHx  - shift + 8.80146457125193);


	for (size_t iter = 0; iter < max_iter; ++iter){
		//update phi
		if (iter == 5)
			write_to_disc("../data/start_vevtor_H2O_rank_60.tttensor",phi);
		XERUS_LOG(info, "------ Iteration = " << iter);
		XERUS_LOG(info,"Projected Gradient step with PC (non symmetric)");
		begin_time = clock();
		res_tangential.clear();
		res_tangential = Top.localProduct(Hs,id,xHx);

		if (iter > 0)
			res_last_tangential = Top.localProduct(res,id);
		res1 = Top.builtTTTensor(res_tangential);
		res_tangential.clear();
		res_tangential = Top.localProduct(res1,Finv);
		if (iter > 0){
			beta = frob_norm(res_tangential)/frob_norm(res_last_tangential); //Fletcher Reeves update
			//std::vector<Tensor> res_tangential_tmp = res_tangential;                            //Polak-Ribiere
			//add(res_tangential_tmp,res_last_tangential,-1.0);                                   //Polak-Ribiere
			//XERUS_LOG(info,"(G_k,G_k-1) = " << innerprod(res_last_tangential,res_tangential));  //Polak-Ribiere
			//beta = innerprod(res_tangential_tmp,res_tangential)/frob_norm(res_last_tangential); //Polak-Ribiere
			add(res_tangential,res_last_tangential, beta);
		}
		res = Top.builtTTTensor(res_tangential);
		projection_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
//		rFr_tmp = contract_TT(id,res,res1);
		residual = res.frob_norm();

		XERUS_LOG(info,"---Time for get res: " << projection_time<<" sekunden");
//		XERUS_LOG(info,rFr_tmp);
//		alpha = -3.0;
//		phi_tmp = phi - alpha * res;
//		phi_tmp.round(max_rank); //round phi
//		phi_tmp /= phi_tmp.frob_norm(); //normalize
//		xHx_tmp = contract_TT(Hs,phi_tmp,phi_tmp);
//		while (xHx_tmp > xHx - alpha*c_1*rFr_tmp){
//			alpha *=roh;
//			phi_tmp = phi - alpha * res;
//			phi_tmp.round(max_rank); //round phi
//			phi_tmp /= phi_tmp.frob_norm(); //normalize
//			xHx_tmp = contract_TT(Hs,phi_tmp,phi_tmp);
//		}
//		XERUS_LOG(info," alpha             = " <<alpha);
//		XERUS_LOG(info," xHx_tmp           = " <<xHx_tmp);
//		XERUS_LOG(info," xHx               = " <<xHx);
//		XERUS_LOG(info," alpha*c_1*rFr_tmp = " <<alpha*c_1*rFr_tmp);

		begin_time = clock();
		res2 = res;
		res2.round(30);

		rHx = contract_TT(Hs,res2,phi);
		rHr = contract_TT(Hs,res2,res2);
		rx = contract_TT(id,res,phi);
		rr = contract_TT(id,res,res);
		alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
//		alpha = alpha_start;
//		while (true){
			phi_tmp = phi - alpha* res;
			phi_tmp.round(std::vector<size_t>(2*nob-1,max_rank),eps);
			xx_tmp = phi_tmp.frob_norm();
			phi_tmp /= xx_tmp;
			xHx_tmp = contract_TT(Hs,phi_tmp,phi_tmp);
//			if (xHx_tmp < xHx - alpha*c1*residual)
//				break;
//			else
//				alpha *= roh;
//		}
		stepsize_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

		XERUS_LOG(info,"---Time for alpha: " << alpha << ": "  << stepsize_time<<" sekunden");
//		XERUS_LOG(info,"---Time for step: " << time (NULL)-begin_time<<" sekunden");
		phi = phi_tmp;
		//XERUS_LOG(info,"Eigenvalue before retraction " <<std::setprecision(10)<< contract_TT(Hs,phi,phi)/phi.frob_norm() - shift + 8.80146457125193);
//			phi = phi +  res;
//		begin_time = time (NULL);
		Top.update(phi);
		res_last = res;

		begin_time = clock();
		xHx = xHx_tmp;
		xx = xx_tmp;
		result.emplace_back(xHx  - shift + 8.80146457125193);
		XERUS_LOG(info,std::setprecision(10) <<phi.ranks());
		XERUS_LOG(info,std::setprecision(10) <<result);
		eigenvalue_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

		//Write to file
		outfile.open(out_name,std::ios::app);
		outfile <<  (value_t) (clock()-global_time)/ CLOCKS_PER_SEC << "," <<std::setprecision(12) << xHx  - shift + 8.80146457125193 << "," << residual <<"," << projection_time <<"," << stepsize_time <<"," << eigenvalue_time <<  std::endl;
		outfile.close();
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

void write_to_disc(std::string name, TTTensor &x){
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,x,xerus::misc::FileFormat::BINARY);
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

		stack(i1,j2,k1,k2,i3) = stack(j1,j2,j3) * xi(j1,k1,i1) * yi(j3,k2,i3);
		stack(i1,i2,i3) = stack(i1,j2,k1,k2,i3)   * Ai(j2,k1,k2,i2);
	}
	return stack[{0,0,0}];
}




