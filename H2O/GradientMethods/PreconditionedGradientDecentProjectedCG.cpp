#include <xerus.h>
#include "../QMC/classes/tangentialOperation.cpp"
#include "ALSres.cpp"
#include "basic.cpp"

using namespace xerus;
using xerus::misc::operator<<;
#define build_operator 0

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
void project(std::vector<Tensor>& x,std::vector<Tensor>& y,const std::vector<Tensor>& Q);
TTOperator particleNumberOperator(size_t d);
value_t getParticleNumber(const TTTensor& x);
value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y);
class localProblem;




int main(){
	size_t nob = 24;
	size_t num_elec = 8;
	double shift = 25.0;
	size_t max_iter = 50;
	size_t max_rank = 20;
	Index ii,jj,kk,ll,mm;
	value_t eps = 10e-6;
	bool round = true,round2 = false;
	size_t rank_precon = 1;
	value_t roh = 0.75, c1 = 10e-4, round_val = 0.9;
	value_t alpha_start = 2;
	std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/PreconProjectedCG_rank_" + std::to_string(max_rank) +"_cpu_bench_new.csv";
	if (round)
		out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/PreconProjectedCG_rank_" + std::to_string(max_rank) +"_rd_"+std::to_string(rank_precon)+"_cpu_bench_new.csv";
	value_t nuc = -52.4190597253;
	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading inverse of Fock Operator");
	TTOperator id,Finv;
	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));
	Finv = (-1)*Finv;
	if (round){
		if (rank_precon == 0)
			Finv = id;
		else
			Finv.round(rank_precon);
	}
	for (size_t i = 0; i < 2*nob;++i){
		Finv.component(i)[{0,0,1,0}] = 0;
		Finv.component(i)[{0,1,0,0}] = 0;
		XERUS_LOG(info,Finv.get_component(i));
	}
	XERUS_LOG(info,Finv.ranks());

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
  std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);


	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi,phi_tmp,phi2;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	project(phi,num_elec,2*nob);
	XERUS_LOG(info,phi.ranks());

	XERUS_LOG(info,"--- starting gradient descent ---");
	value_t rHx,rHr,rx,rr,xx,xHx, xHx_tmp,xx_tmp,projection_time,stepsize_time,eigenvalue_time;
	double alpha,beta,residual;
	clock_t begin_time,global_time = clock();


	TTTensor res, res2, res_last;
	std::vector<value_t> result;
	std::vector<Tensor> res_tangential, res_last_tangential;
	TangentialOperation Top(phi);
	std::ofstream outfile;

	outfile.open(out_name);
	outfile.close();

	xx = phi.frob_norm();
	phi /= xx; //normalize
	xHx = contract_TT(Hs,phi,phi);
	result.emplace_back(xHx  - shift + nuc);
	for (size_t iter = 0; iter < max_iter; ++iter){
		//update phi
		XERUS_LOG(info, "------ Iteration = " << iter);
		XERUS_LOG(info,"Projected Gradient step with PC (non symmetric)");
		XERUS_LOG(info,"Particle Number phi " << std::setprecision(13) << getParticleNumber(phi));

		begin_time = clock();
		res_tangential.clear();
		XERUS_LOG(info,Hs.frob_norm());
		XERUS_LOG(info,Finv.frob_norm());
		XERUS_LOG(info,xHx);
		res_tangential = Top.localProduct(Hs,Finv,xHx,true);
		if (iter == 0){
			res = Top.builtTTTensor(res_tangential);
		} else {
			res_last_tangential = Top.localProduct(res,id);
			beta = frob_norm(res_tangential)/frob_norm(res_last_tangential); //Fletcher Reeves update
			XERUS_LOG(info,"Beta = " << beta);
			add(res_tangential,res_last_tangential, beta);
			res = Top.builtTTTensor(res_tangential);
		}
		projection_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

		residual = res.frob_norm();

		XERUS_LOG(info,residual);
		XERUS_LOG(info,"Norm res " << res.frob_norm());

		XERUS_LOG(info,"---Time for get res: " << projection_time<<" sekunden");
		XERUS_LOG(info,"Particle Number res " << std::setprecision(13) << getParticleNumber(res));

		begin_time = clock();
		//project(res,num_elec,2*nob);
		XERUS_LOG(info,"Particle Number res (after projection) " << std::setprecision(13) << getParticleNumber(res));
		XERUS_LOG(info,"\n" << res.ranks());
		res2 = res;
		//phi2 = phi;
//		if (round2)
//			res2.round(round_val);
		//phi2.round(10);
		rHx = contract_TT(Hs,res2,phi);
//		XERUS_LOG(info,"rHx" << rHx);
//		rHx = contract_TT(Hs,res,phi);
//		XERUS_LOG(info,"rHx" << rHx);
		rHr = contract_TT(Hs,res2,res2);
//		XERUS_LOG(info,"rHr" << rHr);
//		rHr = contract_TT(Hs,res,res);
//		XERUS_LOG(info,"rHr" << rHr);
		rx = contract_TT(id,res2,phi);
//		XERUS_LOG(info,"rx" << rx);
		rr = contract_TT(id,res2,res2);
//		XERUS_LOG(info,"rr" << rr);
		alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
//		alpha = alpha_start;
//		while (true){
	//{{alpha = 3.0;
			phi_tmp = phi - alpha* res;
			phi_tmp.round(std::vector<size_t>(2*nob-1,max_rank),eps);
			xx_tmp = phi_tmp.frob_norm();
			phi_tmp /= xx_tmp;
			xHx_tmp = contract_TT(Hs,phi_tmp,phi_tmp);
//			XERUS_LOG(info,xHx_tmp);
//			XERUS_LOG(info,(xHx-alpha*rHx)/(1-rx*alpha));

//			if (xHx_tmp < xHx - alpha*c1*residual)
//				break;
//			else
//				alpha *= roh;
//		}
		stepsize_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
		XERUS_LOG(info,"---Time for alpha: " << alpha << ": "  << stepsize_time<<" sekunden");
		phi = phi_tmp;
		Top.update(phi);
		res_last = res;

		begin_time = clock();
		xHx = xHx_tmp;
		xx = xx_tmp;
		result.emplace_back(xHx  - shift + nuc);
		XERUS_LOG(info,std::setprecision(10) <<result);
		XERUS_LOG(info,std::setprecision(10) <<phi.ranks());
		eigenvalue_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

		//Write to file
		outfile.open(out_name,std::ios::app);
		outfile <<  (value_t) (clock()-global_time)/ CLOCKS_PER_SEC << "," <<std::setprecision(12) << xHx  - shift +nuc << "," << residual <<"," << projection_time <<"," << stepsize_time <<"," << eigenvalue_time <<  std::endl;
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

value_t getParticleNumber(const TTTensor& x){
	Index ii,jj;
	size_t d = x.order();
	auto P = particleNumberOperator(d);

	Tensor pn,nn;
	pn() = P(ii/2,jj/2)*x(ii&0)*x(jj&0);
	nn() = x(ii&0)*x(ii&0);
	return pn[0]/nn[0];
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

void project(TTTensor &phi, size_t p, size_t d){
	Index i1,i2,j1,j2,k1,k2;
	auto ranks = phi.ranks();
	for (size_t k = 0; k <= d; ++k){

		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t) p - (value_t) k;
			phi /=  f;
			phi.round(1e-12);
			phi /= phi.frob_norm();

		}
	}
	//phi.round(ranks);
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


