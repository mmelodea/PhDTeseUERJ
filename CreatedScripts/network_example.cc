///sigmoid output function
double sigmoid_272(double z){
	return 1./(1+TMath::Exp(-z));
}

///max function (ReLU)
double relu_272(double value){
	double max_val = (value > 0)? value:0;
	return max_val;
}


///PReLU function
double prelu_272(double slope, double value){
	double max_val = (value > 0)? value:value*slope;
	return max_val;
}


///Set of neuron functions with their respective weight and bias
double l5_func_272(std::vector<double> &inputs, int neuron){
	int n_inputs = inputs.size();
	double z = 0;
	switch(neuron){
		case 0:
			z += -0.0070145228*inputs[0];
			z += 0.1580473483*inputs[1];
			z += -0.1274410635*inputs[2];
			z += -0.9732300043;
		break;
	}

	return z;
}
double l4_func_272(std::vector<double> &inputs, int neuron){
	int n_inputs = inputs.size();
	double z = 0;
	switch(neuron){
		case 0:
			z += -0.0238157697*inputs[0];
			z += -0.0651986673*inputs[1];
			z += -0.0350731760*inputs[2];
			z += -0.0821869150;
		break;
		case 1:
			z += -0.1730548739*inputs[0];
			z += 0.2031890899*inputs[1];
			z += -0.0251056589*inputs[2];
			z += 2.8852570057;
		break;
		case 2:
			z += 0.0533006750*inputs[0];
			z += -0.1700736433*inputs[1];
			z += -0.2899119854*inputs[2];
			z += 9.8609056473;
		break;
	}

	return z;
}
double l3_func_272(std::vector<double> &inputs, int neuron){
	int n_inputs = inputs.size();
	double z = 0;
	switch(neuron){
		case 0:
			z += -0.0047135279*inputs[0];
			z += -0.0590741038*inputs[1];
			z += -0.2453326881*inputs[2];
			z += 18.8774299622;
		break;
		case 1:
			z += 0.6622967720*inputs[0];
			z += -0.0050368067*inputs[1];
			z += -0.6060327291*inputs[2];
			z += -7.8763937950;
		break;
		case 2:
			z += -0.7710984349*inputs[0];
			z += 0.0001440368*inputs[1];
			z += 0.5693975687*inputs[2];
			z += 3.5497217178;
		break;
	}

	return z;
}
double l2_func_272(std::vector<double> &inputs, int neuron){
	int n_inputs = inputs.size();
	double z = 0;
	switch(neuron){
		case 0:
			z += 1.2605808973*inputs[0];
			z += 0.4494626522*inputs[1];
			z += 0.0182071533*inputs[2];
			z += -0.4576845169*inputs[3];
			z += 0.5461030006*inputs[4];
			z += 0.0015221462;
		break;
		case 1:
			z += -0.0123674190*inputs[0];
			z += -0.0751453787*inputs[1];
			z += 0.0217242241*inputs[2];
			z += -0.0809679255*inputs[3];
			z += -0.0286572557*inputs[4];
			z += -0.0440520123;
		break;
		case 2:
			z += 1.4803857803*inputs[0];
			z += -0.3530028462*inputs[1];
			z += -0.0238832049*inputs[2];
			z += 0.2339471430*inputs[3];
			z += -0.3269425929*inputs[4];
			z += -29.1441612244;
		break;
	}

	return z;
}
double l1_func_272(std::vector<double> &inputs, int neuron){
	int n_inputs = inputs.size();
	double z = 0;
	switch(neuron){
		case 0:
			z += -0.1659461707*inputs[0];
			z += -0.6928408146*inputs[1];
			z += -0.1003256962*inputs[2];
			z += 0.0585420057*inputs[3];
			z += 0.5957670808*inputs[4];
			z += -1.2804640532*inputs[5];
			z += -0.3506409526*inputs[6];
			z += -1.7032202482*inputs[7];
			z += -0.5168685317*inputs[8];
			z += 0.1162516549*inputs[9];
			z += -1.3009557724*inputs[10];
			z += -0.3004784286*inputs[11];
			z += 0.5537352562*inputs[12];
			z += 2.2581534386*inputs[13];
			z += -0.8218715191*inputs[14];
			z += 1.3330466747*inputs[15];
			z += 0.6919518113*inputs[16];
			z += -0.0395321809*inputs[17];
			z += -42.7089881897;
		break;
		case 1:
			z += -0.4189539552*inputs[0];
			z += -14.1900205612*inputs[1];
			z += -0.5345069766*inputs[2];
			z += -0.3208637536*inputs[3];
			z += -3.8663918972*inputs[4];
			z += 1.4552786350*inputs[5];
			z += -0.4277385771*inputs[6];
			z += -6.0704765320*inputs[7];
			z += 0.7751671672*inputs[8];
			z += -0.1476003975*inputs[9];
			z += -1.9570207596*inputs[10];
			z += 0.8638247252*inputs[11];
			z += 0.1884074658*inputs[12];
			z += 53.1521492004*inputs[13];
			z += 0.3087227643*inputs[14];
			z += 0.3526735306*inputs[15];
			z += -24.0963535309*inputs[16];
			z += -0.9317988157*inputs[17];
			z += -3.8823347092;
		break;
		case 2:
			z += -0.0177176557*inputs[0];
			z += -0.0515748933*inputs[1];
			z += -0.0040869527*inputs[2];
			z += -0.0164461136*inputs[3];
			z += -0.0329781808*inputs[4];
			z += 0.0188262668*inputs[5];
			z += -0.0243709497*inputs[6];
			z += -0.0215751510*inputs[7];
			z += 0.0115934750*inputs[8];
			z += 0.0289968979*inputs[9];
			z += 0.0064768801*inputs[10];
			z += 0.0124688204*inputs[11];
			z += -0.0562790483*inputs[12];
			z += 0.0343059003*inputs[13];
			z += -0.0088156713*inputs[14];
			z += -0.0272380654*inputs[15];
			z += 0.0048469868*inputs[16];
			z += 0.0193189867*inputs[17];
			z += -0.0094796121;
		break;
		case 3:
			z += -0.0688713416*inputs[0];
			z += -32.3510551453*inputs[1];
			z += -1.6043983698*inputs[2];
			z += -0.1478997916*inputs[3];
			z += -12.7351293564*inputs[4];
			z += 0.6689591408*inputs[5];
			z += 0.0226605330*inputs[6];
			z += -13.8519487381*inputs[7];
			z += 0.5693590641*inputs[8];
			z += 0.2408901751*inputs[9];
			z += -5.2771420479*inputs[10];
			z += 0.2275228947*inputs[11];
			z += -0.0637366846*inputs[12];
			z += 32.9030456543*inputs[13];
			z += 0.0465568677*inputs[14];
			z += -0.1250014454*inputs[15];
			z += 32.2035713196*inputs[16];
			z += -1.3601799011*inputs[17];
			z += 17.6616573334;
		break;
		case 4:
			z += -0.3727072477*inputs[0];
			z += -13.9514837265*inputs[1];
			z += -2.1945214272*inputs[2];
			z += -0.3466695249*inputs[3];
			z += -3.2665011883*inputs[4];
			z += 0.8901898265*inputs[5];
			z += -0.4572418928*inputs[6];
			z += -3.8535449505*inputs[7];
			z += -0.1235373020*inputs[8];
			z += -0.1127690673*inputs[9];
			z += -1.9079580307*inputs[10];
			z += -0.1441542804*inputs[11];
			z += 0.1942518502*inputs[12];
			z += -25.9085960388*inputs[13];
			z += -0.1471518874*inputs[14];
			z += 0.3506547809*inputs[15];
			z += 46.9425735474*inputs[16];
			z += -0.5447452664*inputs[17];
			z += -3.7183654308;
		break;
	}

	return z;
}



double dnn_272(std::vector<double> &inputs){
	std::vector<double> l1_neurons_z_272;
	for(int l1_n=0; l1_n<5; l1_n++)
		l1_neurons_z_272.push_back( relu_272(l1_func_272(inputs,l1_n)) );
//------ end layer 1 ------


	std::vector<double> l2_neurons_z_272;
	for(int l2_n=0; l2_n<3; l2_n++)
		l2_neurons_z_272.push_back( relu_272(l2_func_272(l1_neurons_z_272,l2_n)) );
//------ end layer 2 ------


	std::vector<double> l3_neurons_z_272;
	for(int l3_n=0; l3_n<3; l3_n++)
		l3_neurons_z_272.push_back( relu_272(l3_func_272(l2_neurons_z_272,l3_n)) );
//------ end layer 3 ------


	std::vector<double> l4_neurons_z_272;
	for(int l4_n=0; l4_n<3; l4_n++)
		l4_neurons_z_272.push_back( relu_272(l4_func_272(l3_neurons_z_272,l4_n)) );
//------ end layer 4 ------



	return sigmoid_272( l5_func_272(l4_neurons_z_272,0) );
}
