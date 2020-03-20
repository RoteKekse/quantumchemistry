#include <xerus.h>
using namespace xerus;
using xerus::misc::operator<<;



int main(){

	TTOperator Q;
	std::string name = "../data/Q_61_61.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,Q,xerus::misc::FileFormat::BINARY);
	read.close();
	Q.require_correct_format();
	XERUS_LOG(info, Q.ranks());
	Q.round(10e-14);
	XERUS_LOG(info, Q.ranks());



	return 0;
}
