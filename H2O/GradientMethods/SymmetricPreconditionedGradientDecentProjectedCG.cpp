#include <xerus.h>
#include "tangentialOperation.cpp"
#include "ALSres.cpp"
#include "basic.cpp"

using namespace xerus;
using xerus::misc::operator<<;
#define build_operator 0

double get_stepsize(double l, double a, double b, double c, double d);
void write_to_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTOperator &op);
void read_from_disc(std::string name, TTTensor &x);
void project(std::vector<Tensor>& x,std::vector<Tensor>& y,const std::vector<Tensor>& Q);
value_t contract_TT(const TTOperator& A, const TTTensor& x, const TTTensor& y);
value_t contract_TT(const TTOperator& A,const TTOperator& F, const TTTensor& x, const TTTensor& y);
class localProblem;




int main(){
	size_t nob = 25;
	size_t num_elec = 10;
	double shift = 86.0;
	size_t max_iter = 100;
	size_t max_rank = 60;
	Index ii,jj,kk,ll,mm;
	value_t eps = 10e-6;
	value_t roh = 0.75, c1 = 10e-4;
	value_t alpha_start = 2;
	std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/SymPreconProjectedCG_rank_" \
			+ std::to_string(max_rank) +"_cpu_exact3.csv";


	// Load operators
	XERUS_LOG(info, "--- Loading operators ---");
	XERUS_LOG(info, "Loading inverse of Fock Operator");
	TTOperator id,Finv,Finv_old,Fsqinv(std::vector<size_t>(4*nob,2)),Fsq(std::vector<size_t>(4*nob,2)),test;
	Index i1,i2,i3,i4,j1,j2,k1,k2;
	std::string name = "../data/fock_h2o_inv_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name,Finv);
	id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));
	Finv = (-1)*Finv;

	Finv.round(1);
	for (size_t i = 0; i < 2*nob; ++i){
		Tensor tmp({1,2,2,1}),tmp1({1,2,2,1}),tmp2;
		tmp[{0,0,0,0}] = std::sqrt(std::abs(Finv.component(i)[{0,0,0,0}]));
		tmp[{0,1,1,0}] = std::sqrt(std::abs(Finv.component(i)[{0,1,1,0}]));
		tmp1[{0,0,0,0}] = 1/Finv.component(i)[{0,0,0,0}];
		tmp1[{0,1,1,0}] = 1/Finv.component(i)[{0,1,1,0}];
		Fsqinv.set_component(i,tmp);
		Fsq.set_component(i,tmp1);
	}
	Fsqinv.move_core(0);
	Fsq.move_core(0);
	Finv_old = Finv;
	Finv(i1^(2*nob),j1^(2*nob)) = Fsqinv(i1^(2*nob),k1^(2*nob)) *  Fsqinv(k1^(2*nob),j1^(2*nob));
	Finv.move_core(0);
	test = Finv-Finv_old;
	XERUS_LOG(info,"error = " << (test).frob_norm());


	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
  std::string name2 = "../data/hamiltonian_h2o_shifted" + std::to_string(2*nob) +"_full.ttoperator";
	read_from_disc(name2,Hs);
	Hs(i1^(2*nob),j1^(2*nob)) = Fsqinv(i1^(2*nob),k1^(2*nob)) *  Hs(k1^(2*nob),k2^(2*nob))*Fsqinv(k2^(2*nob),j1^(2*nob));

	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor x,x_tmp;
	read_from_disc("../data/eigenvector_H2O_50_-84.880839_3.tttensor",x);
	x(i1&0) = Fsq(i1^(2*nob),j1^(2*nob)) * x(j1&0);
	x.move_core(0);
  XERUS_LOG(info,"--- starting gradient descent ---");
	XERUS_LOG(info,"Projected Gradient step with PC ( symmetric)");
	value_t a,b,c,d,m,n,pHx,pHp,pFx,xHx,xFx,pFp,gFg_new,gFg_old,projection_time,stepsize_time,eigenvalue_time;
	double alpha,beta,residual,residual_new, residual_old;
  clock_t begin_time,global_time = clock();


	TTTensor p, g, g_old;
	std::vector<value_t> result;
	std::vector<Tensor> g_tang, p_tang,x_tang,g_old_tang;
  std::ofstream outfile;
	outfile.open(out_name);
	outfile.close();

	m = contract_TT(Finv,x,x);
	x /= std::sqrt(m); //normalize
	n = contract_TT(Hs,x,x);
	XERUS_LOG(info,n);

	TangentialOperation Top(x);
	g_tang = Top.localProductSymmetric(Hs,Finv,n); //check this
	g = Top.builtTTTensor(g_tang);
	p = g;
	p_tang = g_tang;

	result.emplace_back(n  - shift + 8.80146457125193);



	for (size_t iter = 0; iter < max_iter; ++iter){
		XERUS_LOG(info, "------ Iteration = " << iter);
		begin_time = clock();
		//p = Top.builtTTTensor(p_tang);
		a = contract_TT(Hs,p,x);
		b = contract_TT(Hs,p,p);
		c = contract_TT(Finv,p,x);
		d = contract_TT(Finv,p,p);
		alpha = get_stepsize(n,a,b,c,d);
		stepsize_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
		XERUS_LOG(info,"---Time for alpha: " << alpha << ": "  << stepsize_time<<" sekunden");


		begin_time = clock();
//		x_tang  = Top.localProduct(x,id);
//		add(x_tang,p_tang, alpha);
//		x = Top.builtTTTensor(x_tang);
		x = x + alpha*p;
		x.round(std::vector<size_t>(2*nob-1,max_rank),eps);

		m = contract_TT(Finv,x,x);
		x /= std::sqrt(m); //normalize
		n = contract_TT(Hs,x,x);

		result.emplace_back(n  - shift + 8.80146457125193);
		XERUS_LOG(info,std::setprecision(10) <<result);
		XERUS_LOG(info,std::setprecision(10) <<x.ranks());
		eigenvalue_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
		XERUS_LOG(info,"---Time for get EV: " << eigenvalue_time<<" sekunden");



		begin_time = clock();
		Top.update(x);
		p_tang = Top.localProduct(p,id);
		g_old_tang = Top.localProduct(g,id);
		g_tang = Top.localProductSymmetric(Hs,Finv,n);
		g_old = Top.builtTTTensor(g_old_tang);
		g = Top.builtTTTensor(g_tang);
		p = Top.builtTTTensor(p_tang);
		XERUS_LOG(info,"x^Tg = " << contract_TT(id,g,x));
		XERUS_LOG(info,"p^Tg = " << contract_TT(id,g,p));
		beta = (-1)*contract_TT(Hs,g,p) / b;
		//beta = contract_TT(id,g,g-g_old)/contract_TT(id,g_old,g_old);

		XERUS_LOG(info,"beta    = " << beta);
		XERUS_LOG(info,"g       = " <<contract_TT(id,g,g));
		XERUS_LOG(info,"g,g_old = " <<contract_TT(id,g,g_old));
		XERUS_LOG(info,"g_old   = " <<contract_TT(id,g_old,g_old));
		//add2(p_tang,g_tang, beta);
		p = g + beta*p;
		//p_tang = Top.localProduct(p,id);

		projection_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
		XERUS_LOG(info,"---Time for get res: " << projection_time<<" sekunden");



		//Write to file
		outfile.open(out_name,std::ios::app);
		outfile <<  (value_t) (clock()-global_time)/ CLOCKS_PER_SEC << "," <<std::setprecision(12) << n  - shift + 8.80146457125193 << "," << residual <<"," << projection_time <<"," << stepsize_time <<"," << eigenvalue_time <<  std::endl;
		outfile.close();

	}
	return 0;
}

double get_stepsize(double n, double a, double b, double c, double d){
	double delta = std::pow((n*d-b),2) - 4*(b*c-a*d)*(a-n*c);
	double alpha = (n*d - b + std::sqrt(delta))/(2*(b*c-a*d));
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
	return stack[{0,0,0}];
}

value_t contract_TT(const TTOperator& A,const TTOperator& F, const TTTensor& x, const TTTensor& y){
	Tensor stack = Tensor::ones({1,1,1,1,1});
	size_t d = A.order()/2;
	Index i1,i2,i3,i4,i5,j1,j2,j3,j4,j5,k1,k2,k3,k4;
	for (size_t i = 0; i < d; ++i){
		auto Ai = A.get_component(i);
		auto Fi = F.get_component(i);
		auto xi = x.get_component(i);
		auto yi = y.get_component(i);

		stack(i1,i2,i3,i4,i5) = stack(j1,j2,j3,j4,j5) * xi(j1,k1,i1) * Fi(j2,k1,k2,i2) * Ai(j3,k2,k3,i3) * Fi(j4,k3,k4,i4) * yi(j5,k4,i5);
	}
	return stack[{0,0,0}];
}




