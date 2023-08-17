clc;
clear all;
%PeColi,sequenceColi加载eColi.csv

[P2eColi,PepeColi,VarName3] = importfile('E:\CYNTHIA\CNYresearch\论文7\Glutarylation\PepExtract\P1.csv');   %正数据集，包含accession号，peptide，以及修饰位置
[PeColi,sequenceColi] = importfile1('E:\CYNTHIA\CNYresearch\论文7\Glutarylation\PepExtract\Glufasta.csv');  %保存accession号，和氨基酸序列
prefix=('E:\CYNTHIA\CNYresearch\论文7\Glutarylation\Glufeature\feature\glutarylation208\pssm');
d=dir([prefix]);
ll=8;
nn=0;
sum=0;
kkk=0;
for i=3:length(d)
    filename=['E:\CYNTHIA\CNYresearch\论文7\Glutarylation\Glufeature\feature\glutarylation208\pssm\',d(i).name];
    test1=importdata(filename);
    textdata=test1.textdata;
    [m,n]=size(textdata);
    fasta=[];
    for j=3:m
        if length(char(textdata(j,2)))==1
            fasta=strcat(fasta,char(textdata(j,2)));
        end
    end
    m1=length(fasta);
    data=test1.data(1:m1,1:20);
    [p,q]=size(sequenceColi);
    for k=1:p   %p为577个氨基酸
        pp=0;
        if strcmp(fasta,char(sequenceColi(k)))==1
           mm=find(strcmp(P2eColi,char(PeColi(k))));
           pp=pp+1;
           if pp==2
               print pp
           end
           aa=VarName3(mm);
           m2=length(aa);
           kkk=kkk+1;
           ff{kkk,1}=fasta;
           sum=sum+m2;
           for kk=1:m2
              pp=aa(kk);
              nn=nn+1;
              if pp+ll>m1
                  peptide{nn,1}=[data((pp-ll):end,:);zeros(pp+ll-m1,20)];
              elseif pp-ll<1
                  peptide{nn,1}=[zeros(ll-pp+1,20);data(1:(pp+ll),:)];
              else
                  peptide{nn,1}=data((pp-ll):(pp+ll),:);
              end
           end
        end
    end
end


save PssmP.mat peptide