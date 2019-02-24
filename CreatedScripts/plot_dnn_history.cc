#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <TLorentzVector.h>
#include <TFile.h>
#include <TTree.h>
#include <TString.h>


void plot_dnn_history(void){

  ifstream Input("/home/mmelodea/CernBox/Mine/LXPLUS/KerasTraining/original/test10/log_file.log");
  
  TGraph *gloss = new TGraph();
  TGraph *gacc = new TGraph();
  TGraph *gvloss = new TGraph();
  TGraph *gvacc = new TGraph();
  
  std::string info, info_1 = "aaa";
  float loss;
  float val_loss;
  float acc;
  float val_acc;
  int epoch;
  //int line = 0;
  while(info != "Accuracy"){
    Input >> info;
    if(info_1 == "Epoch") epoch = std::stof(info);
    if(info_1 == "loss:"){
      //std::cout<<"\nloss: "<<info;
      loss = std::stof(info);
      gloss->SetPoint(epoch-1,epoch,loss);
    }
    if(info_1 == "acc:"){
      //std::cout<<"\tacc: "<<info;
      acc = std::stof(info);
      gacc->SetPoint(epoch-1,epoch,acc);
    }
    if(info_1 == "val_loss:"){
      //std::cout<<"\tval_loss: "<<info;
      val_loss = std::stof(info);
      gvloss->SetPoint(epoch-1,epoch,val_loss);
    }
    if(info_1 == "val_acc:"){
      //std::cout<<"\tval_acc: "<<info<<std::endl;
      val_acc = std::stof(info);
      gvacc->SetPoint(epoch-1,epoch,val_acc);
    }
    
    info_1 = info;
    //++line;
    //if(line > 100) break;
  }
  
  gloss->SetLineWidth(2);
  gloss->SetLineColor(kBlack);
  gacc->SetLineWidth(2);
  gacc->SetLineColor(kRed);
  gvloss->SetLineWidth(2);
  gvloss->SetLineColor(kBlue);
  gvacc->SetLineWidth(2);
  gvacc->SetLineColor(kViolet);
  TMultiGraph *mg = new TMultiGraph();
  mg->Add(gloss);
  mg->Add(gacc);
  mg->Add(gvloss);
  mg->Add(gvacc);
  
  TCanvas *cv = new TCanvas("cv","cv",10,10,700,700);
  mg->Draw("al");
  mg->GetXaxis()->SetTitle("Epochs");
  cv->SetLogx();
  
  TLegend *leg = new TLegend(0.6,0.5,0.8,0.7);
  leg->SetFillColor(0);
  leg->SetBorderSize(0);
  leg->SetTextSize(0.05);
  leg->AddEntry(gloss,"loss","l");
  leg->AddEntry(gvloss,"val_loss","l");
  leg->AddEntry(gacc,"acc","l");
  leg->AddEntry(gvacc,"val_acc","l");
  leg->Draw();
}
