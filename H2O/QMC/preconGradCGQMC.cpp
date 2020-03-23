#include <xerus.h>

#include "../loading_tensors.cpp"
#include "classes_old/basic.cpp"
#include "classes_old/tangential.cpp"
#include "classes_old/tangentialOperation.cpp"

TTOperator particleNumberOperator(size_t k, size_t d);
void project(TTTensor &phi, size_t p, size_t d);


int main(){
	size_t nob = 24,num_elec = 8;
	value_t nuc = -52.4190597253,ref = -76.25663396, shift = 25.0, beta,alpha,eps = 1e-6;
	std::string path_T = "../data/T_H2O_48_bench.tensor";
	std::string path_V= "../data/V_H2O_48_bench.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };


	size_t max_iter = 1, max_rank = 20, iterations = 1e6;

	std::string out_name = "/homes/numerik/goette/Documents/jupyter_examples/Paper/Preconditioning/data/preconGradCGQMC_rank_" +
			std::to_string(max_rank) +"_cpu_bench.csv";

	XERUS_LOG(info, "--- Initializing Start Vector ---");
	XERUS_LOG(info, "Setting Startvector");
	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	project(phi,num_elec,2*nob);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
  std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	std::ofstream outfile;
	outfile.open(out_name);
	outfile.close();

	Tangential tang(2*nob,num_elec,iterations,path_T,path_V,shift,sample,phi);
	TangentialOperation Top(phi);

	value_t ev = tang.get_eigenvalue();
	XERUS_LOG(info, "Eigenvalue start: " << ev);
	XERUS_LOG(info, "Eigenvalue exact start: " << contract_TT(Hs,phi,phi));

	clock_t begin_time,global_time = clock();
	std::vector<Tensor> res_tangential, res_last_tangential;
	value_t projection_time,stepsize_time,eigenvalue_time,residual;
	for (size_t iter = 0; iter < max_iter; ++iter){
		XERUS_LOG(info, "------ Iteration = " << iter);
		begin_time = clock();
		res_tangential.clear();
		res_tangential = tang.get_tangential_components(ev);
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
		XERUS_LOG(info,"---Time for get res: " << projection_time<<" sekunden");

		begin_time = clock();
		alpha = 3.0;
		phi = phi - alpha* res;
		phi.round(std::vector<size_t>(2*nob-1,max_rank),eps);
		phi /= phi.frob_norm();

		project(phi,num_elec,2*nob);

		name2 = "../data/residual_" + std::to_string(2*nob)+"_"+ std::to_string(max_rank)  +"_benchmark.tttensor";
		std::ofstream write(name2.c_str() );
		xerus::misc::stream_writer(write,res,xerus::misc::FileFormat::BINARY);
		write.close();

		tang.update(phi);
		ev = tang.get_eigenvalue();
		XERUS_LOG(info, "Eigenvalue: " << ev);
		XERUS_LOG(info, "Eigenvalue exact: " << contract_TT(Hs,phi,phi));
		stepsize_time = (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
		XERUS_LOG(info,"---Time for alpha: " << alpha << ": "  << stepsize_time<<" sekunden");

		Top.update(phi);
		res_last = res;

		//Write to file
		outfile.open(out_name,std::ios::app);
		outfile <<  (value_t) (clock()-global_time)/ CLOCKS_PER_SEC << "," <<std::setprecision(12) << ev  - shift +nuc << "," << residual <<"," << projection_time <<"," << stepsize_time <<"," << eigenvalue_time <<  std::endl;
		outfile.close();

	}



	return 0;
}

void project(TTTensor &phi, size_t p, size_t d){
	Index i1,i2,j1,j2,k1,k2;
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
	for (size_t k = 0; k <= d; ++k){
		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t)p - (value_t) k;
			phi /=  f;
			phi.round(1e-12);
		}
	}
	phi.round(1e-4);
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
	phi /= phi.frob_norm();
	XERUS_LOG(info,phi.ranks());
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


