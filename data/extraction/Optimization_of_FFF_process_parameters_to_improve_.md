Received: 30 April 2024 Revised: 18 June 2024 Accepted: 15 July 2024 

DOI: 10.1002/pen.26900 

## R E S E A R C H A R T I C L E 

## Optimization of FFF process parameters to improve the tensile strength and impact energy of polylactic acid/carbon nanotube composite 

Hatam Hardani[1] | Mahmoud Afshari[2] | Fatemeh Allahyari[3] | Mohammad Reza Samadi[4] | Hossein Afshari[5] | Edison Marcelo Melendres Medina[6] 

1Department of Mechanical Engineering, Ahvaz Branch, Islamic Azad University, Ahvaz, Iran 

2Department of Mechanical Engineering, Amirkabir University of Technology, Tehran, Iran 

3Faculty of Sciences, Arak University, Arak, Iran 

4Faculty of Mechanical Engineering, Technical and Vocational University 

(TVU), Tehran, Iran 

5Mechanical Engineering Department, Engineering Faculty, University of Birjand, Birjand, Iran 

6Facultad de Administracion de� Empresas, Escuela Superior Politécnica de Chimborazo (ESPOCH), Riobamba, Ecuador 

## Correspondence 

Mahmoud Afshari, Department of Mechanical Engineering, Amirkabir University of Technology, Tehran, Iran. Email: m.afshari@aut.ac.ir 

## Abstract 

In this study, the process parameters of fused filament fabrication are optimized to improve the tensile strength and impact energy of polylactic acid/carbon nanotube (PLA/CNT) composite. Hence, the utility function (UF) technique and response surface method (RSM) are applied to explore the optimal levels of the effective parameters of print speed, nozzle temperature, and CNT content. The differential scanning calorimetry (DSC), thermogravimetric analysis (TGA), and SEM analyses are employed to examine the thermal properties of the printed samples. The results of DSC and TGA analyses exhibited that the incorporation of CNT into PLA enhanced the thermal strength of PLA/CNT composite. The addition of CNTs in the composite improved the tensile strength by 37%, while the addition of CNTs up to 4 wt% improved the impact energy by 29%. Moreover, an increment of the print speed to 60 mm/s reduced the impact energy (12%) and tensile strength (22%), while an increment of the nozzle temperature to 200[�] C enhanced the impact energy (9%) and tensile strength (12%). The optimization results demonstrated that the strength and impact energy of PLA/CNT composite optimized at CNT content of 2.8 wt%, print speed of 20 mm/s, and nozzle temperature of 209[�] C. Additionally, the impact energy and tensile strength of the PLA/CNT composite enhanced up to 62.5 MPa and 2.14 J at the optimum conditions. 

## Highlights 

- Application of FFF process for producing the PLA/CNT composite 

- Investigating the impact of FFF parameters on the mechanical properties 

- Estimating the optimal conditions of the FFF process 

## K E Y W O R D S 

impact energy, optimization, PLA/CNT composite, tensile strength 

© 2024 Society of Plastics Engineers. 5047 

wileyonlinelibrary.com/journal/pen 

Polym Eng Sci. 2024;64:5047–5060. 

HARDANI ET AL. 

5048 

## 1 | INTRODUCTION 

Polylactic acid (PLA) is a thermoplastic polymer that has excellent biocompatibility, good biodegradability, high mechanical strength, and suitable processability.[1] This polymer is widely used in the packaging industry, food industry, and various medical applications like drug delivery systems, tissue engineering, membranes, implants, scaffolds, and sutures.[2][–][4] According to the mentioned properties of PLA, it has great potential in the improvement of nano-composite materials.[5] Despite the numerous advantages of PLA, its inherent brittleness and low thermal stability have restricted its extensive applications.[6,7] Thus, an additive is needed to ensure the stiffness of the PLA matrix.[8] One of the excellent additives that simultaneously improves the mechanical properties of polymers is carbon nanotubes (CNTs).[8,9] Previous studies have well established that carbon nanotubes have a good potential for application in different industries due to their excellent characteristics such as good mechanical, electrical, and thermal stabilities.[10] Carbon nanotubes are typically added to composites in small amounts because carbon fibers are expensive; however, they indicate good mechanical behavior.[11] Composites reinforced with a small amount of CNT (2 wt %) can be used in the automotive industry, such as body parts, power strain materials, exhaust systems, engines, breaking systems, suspension, catalytic converters, and electronic equipment.[11,12] Yang et al.[13] reported that by adding 6 wt% CNT content, the tensile strength and bending resistance of PLA/CNT improved by about 64.12% and 29.29%, respectively. Zhou et al.[6] stated that the elongation, tensile strength, and impact energy of PLA enhanced by addition of CNTs. They also stated that the thermal stability of PLA/CNTs composite increased than pure PLA. Vidakis et al.[8] indicated that the mechanical features of PLA/CNT composite are considerably developed by incorporation of 1 wt% CNTs. Pinto et al.[9,14] have also found that the mechanical features of PLA/CNT composite were enhanced by incorporation of CNTs. Although the use of CNTs improves the mechanical behavior of the PLA/CNT composite, but excessive amounts of CNTs lead to the aggregation of the nanoparticles. Therefore, the optimal amount of CNTs should be determined. 

Recently, great attention has been attracted to the fabrication of the polymer composites using 3D printers.[15][–][17] One of the simplest and cheapest techniques for fabrication of polymer composites is the fused filament fabrication (FFF) method. In this process, a polymer filament is placed in a heated nozzle to melt. The nozzle injects the semi-solid filament onto a worktable to create the desired 3D part with deposited layers.[18] Usually, the mechanical strength and manufacturing quality of the sample produced by the FFF method are lower than the sample produced by the 

traditional methods, which is owing to the formation of successive layers in the process and the anisotropy of the microstructure.[13] Therefore, it is required to enhance the mechanical behavior and manufacturing quality of the produced samples by optimizing the process parameters. Different studies have been done to increase the quality of printed samples and improve the properties of the produced parts. To improve the performance and quality of the printed samples, several parameters should be controlled.[19] Anis et al.[20] reported that the nozzle temperature and print speed significantly influence the dimensional accuracy of printed PLA parts. Mohammadimehr et al.[21] stated that a reduction in the thickness of filament layer increases the surface quality and dimensional accuracy of the printed PLA, while a reduction in the print speed leads to an improvement in the mechanical properties. Chacon et al.[22] concluded that the increase in thickness of filament layer and the decrease in print speed improved the mechanical strength of PLA. Khosravani et al.[23] reported that the higher tensile strength and Young's modulus are obtained with the print speed of 20 mm/s; however, Kuznetsov et al.[24] stated that higher tensile strength have attained with the print speed of 25 mm/s at temperature of 210[�] C. In another studies, Altan et al.[25] found that a high strength PLA attained with the print speed of 80 mm/s; however, Cardoso et al.[26] observed that the lower print speed increased the strength of PLA. Benwood et al.[27] have also demonstrated that a rise of bed and nozzle temperatures enhanced the impact energy, Young's modulus, tensile strength, and bending resistance of PLA. The changes of mechanical properties of PLA during the 3D printing process were also examined by Hsueh et al.,[28] Wach et al.,[29] Liparoti et al.,[30] Vanaei et al.,[31] and Deshwal et al.[32] 

A literature review indicated that the 3D print process of PLA has been widely studied by previous researchers. However, the fused filament fabrication of PLA/CNT composite has not been implemented previously. Therefore, the first innovation of this research is that the PLA/CNT composite was fabricated for the first time by using the FFF process. The second innovation of this research is to examine the microstructure and thermal behavior of the fabricated composite by differential scanning calorimetry (DSC), thermogravimetric analysis (TGA), and SEM analyses. Nevertheless, there are many problems in the fabrication of PLA/CNT composites, which require accurate choice of the process parameter range. Thus, the third innovation of this research is to investigate the influence of the CNT content, print speed, and nozzle temperature on the tensile strength and impact energy of the composite using the response surface method (RSM). The latest innovation of this research is to explore the optimal values of the FFF parameters for simultaneously enhancing the impact energy and tensile strength of the PLA/CNT composite using the utility function (UF) method. 

HARDANI ET AL. 

5049 

## 2 | EXPERIMENTAL WORKS 

## 2.1 | Materials 

The raw materials were prepared in powder form to produce the composites. The PLA, used in the present study, was supplied from Plastika Kritis S.A (Heraklion, Crete, Greece). PLA was an industrial grade (PLA-3052D) in the form of a fine powder. The melt flow index, molecular weight, melting temperature, glass transition temperature, and tensile strength of PLA were 14 g/10 min (ASTM D1238), 116,000 g/mol, 153[�] C, 58[�] C, and 62 MPa (ASTM D638), respectively. The CNTs, purchased from INP Corporation, have a purity of more than 90%. The specific surface area, average diameter, and average length of CNTs were 250–300 m[2] /g, 9.5 nm, and 1.5 μm, respectively. The prepared materials were dried in an oven to prepare for mixing in different concentrations. 

## 2.2 | Fused filament fabrication 

In the present research, the PLA/CNT composite was fabricated by using the FFF process. A mechanical mixer is also used to mix CNTs with PLA at different concentrations (0 to 4 wt %). Before the filament extrusion, powder compounds of the PLA/CNT composite are dried in an oven at temperature of 80[�] C. Moreover, the pure PLA powder is dried at temperature of 45[�] C for 24 h. To disperse the CNTs in the polymer, different powder mixtures were fed into the extrusion system. The extrusion of filament was executed at 205[�] C with a rotational velocity of 3.5 rpm. The filament with a diameter of 1.5 mm was fabricated at different concentrations of CNTs by a filament extruder (Filabot EX2, USA). For a better comparison of extruded filaments, pure PLA filament was also fabricated. By studying previous works and performing trial experiments, the ideal conditions of the mixing process and filament extrusion parameters were determined.[1,8] The morphology of filaments was examined visually before performing any test. The fabricated filaments are used for producing the parts in the FFF process. The Solidworks software was also applied to design the print parts. A 3D printer (Model Wanhao D12-230) was used to implement the FFF process and print the composite parts. The different values of nozzle temperature and print speed were applied to perform the FFF process using a 3D printer. Table 1 indicates 

the constant parameters of 3D printer process. Figure 1 shows the experiments performed in the present study to fabricate PLA/CNT composite and investigate its properties. 

## 2.3 | Experimental design 

The RSM is applied in this research to determine the experimental tests and analyze the impact of the FFF parameters on the mechanical characteristics of PLA/CNT composite. To find a relationship between the output responses (tensile strength and impact energy) and the input variables (CNT content, scanning speed, and nozzle temperature), the Box–Behnken design was chosen for RSM technique. The levels of the selected parameters were determined by conducting the primary experiments and investigating the previous works. According to RSM method, 3 levels have been selected for the parameters (Table 2). In the RSM method, 3 center points were considered, and therefore, 15 experiments have been implemented, as given in Table 3. The values of the impact energy and tensile strength have also measured at different conditions of the process parameters and given in Table 3. 

## 2.4 | Mechanical and thermal tests 

The tensile test samples were produced by the FFF process based on ASTM D-638 standard. A Zwick/RoellZ100 machine is used to do the tensile test with a velocity of 50 mm/min at room temperature. Figure 2 displays some broken tensile specimens with the higher tensile strength. The impact specimens were also printed by FFF process based on ASTM D-256. 

The DSC analysis is carried out (NETZCH 200 F3, Maia) to find the crystallization temperature (Tc) and melting temperature (Tm) of the printed samples. The DSC analysis is conducted in a nitrogen environment at temperatures of 20 to 200[�] C with a scan rate of 10[�] C/ min. The heating–cooling–heating cycles are conducted to complete the DSC analysis. First, the specimens are heated from 20 to 200[�] C and kept for 2 min, then cooled to 20[�] C and kept for 2 min, and finally they are reheated up to 200[�] C with a heating rate of 10[�] C/min. For calculating the crystallinity (%) of the printed samples, Equation (1) is used. 

TABLE 1 Constant parameters of FFF process. 

|Parameters|Nozzle diameter|Air gap|Layer thickness|Bed temperature|Built orientation|
|---|---|---|---|---|---|
|Values|0.4 mm|0.05 mm|0.2 mm|80�C|0�|



HARDANI ET AL. 

5050 

**==> picture [370 x 331] intentionally omitted <==**

FIGURE 1 The experiments performed in the present study to fabricate PLA/CNT composite and investigate its properties. 

TABLE 2 The considered levels of the parameters. 

|Parameters|Lower level (�1)|Intermediate level (0)|Higher level (+1)|
|---|---|---|---|
|CNT content (wt%)|0|2|4|
|Print speed (mm/s)|20|40|60|
|Nozzle temperature (�C)|180|200|220|



ΔHm Crystallinityð%Þ ¼ � 100%: ð1Þ �ΔHL � 

In this equation, ΔHL indicates the enthalpy for a perfect crystal of PP (207 J/g) and ΔHm indicates the measured enthalpy of melting. The thermogravimetric analysis (TGA) is also performed for neat PLA and PLA/CNT composite by TA instrument 2960 (New Castle, DE, USA). The thermogravimetric analysis is implemented from 25 to 600[�] C, with a heating rate of 10[�] C/min in nitrogen atmosphere. 

## 2.5 | Analysis of microstructure 

To examine the fracture area and microstructure of tensile specimens, the scanning electron microscope (Model VEGATESCAN, Czech Republic) is used. The fracture 

area of the printed parts is gold-coated to prepare the SEM samples. The SEM images were obtained at 20 kV in a vacuum atmosphere. SEM images were also used to detect the dispersion of the CNTs inside the PLA matrix. 

## 3 | RESULTS AND DISCUSSION 

## 3.1 | Analysis of thermal properties 

The DSC analysis was performed to investigate the thermal properties of PLA/CNT composite. The DSC analysis for PLA/CNT composite was presented in Figure 3. The analysis of the thermal properties of PLA/CNT composite was conducted by the DSC test, as given in Table 4. 

Table 4 indicates that a rise of CNT content to 4 wt% was followed by an increase in the crystallization and 

HARDANI ET AL. 

5051 

||T15|2|40|200|62.03|1.73|
|---|---|---|---|---|---|---|
||T14|2|40|200|61.89|1.66|
||T13|2|40|200|62.48|1.71|
||T12|2|60|220|58.5|1.76|
||T11|2|20|220|59.62|1.98|
||T10|2|60|180|50.27|1.39|
||T9|2|20|180|58.16|1.95|
||T8|4|40|220|51.33|2.01|
||T7|0|40|220|46.29|1.28|
||T6|4|40|180|44.57|1.46|
||T5|0|40|180|39.36|1.3|
|TABLE 3<br>The design matrix of parameters and values of responses.|Tests<br>T1<br>T2<br>T3<br>T4|CNT content (wt%)<br>0<br>4<br>0<br>4|Print speed (mm/s)<br>20<br>20<br>60<br>60|Nozzle temperature (�C)<br>200<br>200<br>200<br>200|Tensile strength (MPa)<br>44.2<br>55.48<br>41.72<br>43.84|Impact energy (J)<br>1.72<br>2.16<br>1.33<br>1.64|



melting temperatures of the composites. By adding the CNTs into PLA matrix, the thermal conductivity of the composite increases. Therefore, the thermal stability and heat transfer of the PLA/CNT composite improve because of the greater thermal conductivity of CNTs compared with PLA.[13,33,34] In addition, DSC results in Table 4 demonstrate that by increasing the amount of CNT to 4 wt%, the crystallinity percentage of the composite improved because CNT nanoparticles acts as nucleation locations for PLA molecules.[35] 

The thermogravimetric analysis was also carried out to analyze the thermal behavior of PLA/CNT composite in temperature range of 0 to 600[�] C, as indicated in Figure 4. From Figure 4, it is apparent that the thermal stability enhanced by adding CNTs into the PLA. When the CNT content is 0 wt%, the PLA matrix begins to thermally degrade at 298[�] C, and mass loss continues to 541[�] C. However, the decomposition temperature of the PLA/CNT composite enhanced up to 311 and 338[�] C by adding the CNT up 2 and 4 wt%. The improvement of the decomposition temperature of PLA/CNT composite is owing to preventing the movement of polymer chains by CNTs.[36,37] It was also reported by other researchers[6,38] that the decomposition temperature of PLA/CNT composite improved by the increase of CNT content. 

The thermal behavior of the composite is analyzed by the weight loss temperature at 10, 50, and 90%, as shown in Table 5. As clear from Table 5, an increase in the CNT content enhanced the thermal stability of PLA. A rise of the decomposition temperature of PLA polymer by the addition of CNTs was also reported in.[36][–][38] 

## 3.2 | Analysis of microstructure 

The interlayer adhesion in the FFF process is very important because the fracture usually happens at the interface of deposited filaments. Thus, the microstructure and mechanical behavior of the samples are analyzed to investigate the adhesive between the deposited layers. The microstructure of tensile samples for different percentages of CNT content was shown in Figure 5. It is obvious from Figure 5 that the nanoparticles were well distributed inside the PLA matrix at CNT content of 2 wt% (Figure 5B). The interfacial interaction of the CNTs and PLA matrix improved under this condition, which tends to increase the mechanical properties of the composite. Nevertheless, a rise in the CNT content to 4 wt% was followed by the aggregation of the nanoparticles (Figure 5C). The aggregation of CNTs leads to the non-uniform distribution of the nanoparticles inside the PLA matrix, which results in the deterioration of the mechanical features of the composite. It should be noted that CNT nanoparticles are not present in Figure 5A. 

HARDANI ET AL. 

5052 

**==> picture [370 x 94] intentionally omitted <==**

FIGURE 2 The broken specimens after tensile test. 

**==> picture [370 x 222] intentionally omitted <==**

FIGURE 3 Analysis of DSC for heating and cooling cycles. 

TABLE 4 Results of thermal analysis for PLA/CNT composite. 

|Thermal properties|PLA+0 wt% CNT|PLA+2 wt% CNT|PLA+4 wt% CNT|
|---|---|---|---|
|Crystallization temperature (�C)|115.4|118.6|120.4|
|Crystallinity (%)|49.6|52.3|54.2|
|Melting temperature (�C)|169.1|171.5|172.7|



Figure 6 presents the microstructure of the PLA/CNT composites achieved at different print speeds. It is apparent from Figure 6A that a good bonding was formed between the filament layers at the lower print speed (20 mm/s) because of higher heat generation. In fact, when the filament is printed with slower speed, it spends more time inside the heated nozzle, which led to a rise in the filament temperature and an improvement in the adhesion bonding of filament layers. When the adhesion between the filament layers increases, the mechanical strength of the composite improves. However, an increase in the speed of printing up to 30 mm/s reduced the bonding between the printed filaments (Figure 6B). Therefore, an increment of the print speed to 30 mm/s was followed by the nucleation of defects in the microstructure and thus the reduction of the mechanical strength of the composite. According to Figure 6C, the presence of cavities and cracks in the fracture surface of the composite increased 

by increasing the speed of print to 40 mm/s due to the decrease of the cooling time and consequently the decrease in the adhesion of printed filaments. 

Figure 7 represents the microstructure of tensile samples at different nozzle temperatures. In Figure 7A, it is evident that the adhesion between the filament layers is low for the nozzle temperature of 180[�] C because the viscosity of the polymer is high. In the other words, the adhesion between the filaments deteriorated at high filament viscosity that produced the cavities and cracks in the fracture surface of the printed samples. In fact, when the viscosity of the filament increases, the fluidity of the material decreases and it does not deposit effectively. Therefore, the adhesion between adjacent filaments is reduced due to the prevention of interpenetration of the polymer chains.[39,40] According to Figure 7B, the adhesion of the filament layers improved with an increase in the temperature of nozzle to 200[�] C because of the high filament viscosity. Thus, the 

HARDANI ET AL. 

5053 

FIGURE 4 Impact of CNT content on thermogravimetric results. 

**==> picture [370 x 240] intentionally omitted <==**

TABLE 5 Thermal stability of the composite at different CNT contents. 

|Temperature|PLA+0 wt% CNT|PLA+2 wt% CNT|PLA+4 wt% CNT|
|---|---|---|---|
|T10(�C)|356|379|393|
|T50(�C)|418|442|455|
|T90(�C)|473|488|501|



**==> picture [500 x 145] intentionally omitted <==**

FIGURE 5 The microstructure of the composite at CNT content of (A) 0, (B) 2, (C) 4 wt% (Magnification: 20000X). 

surface of the printed sample did not showed the crack and cavity at 200[�] C. However, Figure 7C indicates that an increment of the temperature of nozzle to 220[�] C decreased the adhesion bonding between the filaments, which was associated with the increase of cracks in the microstructure. 

## 3.3 | Analysis of tensile strength and impact energy 

The mechanical properties of tensile strength and impact energy were studied by performing the analysis of 

variance (ANOVA). This technique was applied to evaluate the impact of nozzle temperature (NT), CNT content (CNT), and print speed (PS) on the tensile strength (UTS) and impact energy (IE) of PLA/CNT composite. Minitab 17 software is used to perform the ANOVA technique for UTS and IE, as the results were given in Tables 6 and 7, respectively. 

A parameter in the ANOVA table is significant when its P-value is below 0.05. Also, a greater F-value denotes the higher impact of a parameter. From Tables 6 and 7, it can be understood that all parameters of CNT content, PS, and NT have a significant influence on the IE and 

HARDANI ET AL. 

5054 

**==> picture [500 x 129] intentionally omitted <==**

FIGURE 6 Microstructure of the composites at print speed of (A) 20, (B) 30, and (C) 40 mm/s (Magnification: 200X). 

**==> picture [501 x 124] intentionally omitted <==**

FIGURE 7 Microstructure of the composites at nozzle temperature of (A) 180, (B) 200, and (C) 220[�] C (Magnification: 100X). 

UTS. Nevertheless, the F-value for parameters represents that the impact of CNT on the UTS is more than the other parameters, while the PS showed the higher impact on the IE. The regression equations (2) and (3) were also extracted from the analysis of variance for UTS and IE. 

UTS ¼ �279 þ 17:52 CNT – 0:419 PS 

   - þ 3:188 NT – 3:384 CNT � CNT – 0:0057 PS � PS – 0:008 NT � NT – 0:0572 CNT 

   - PS – 0:0042 PS � NT: ð2Þ 

- IE ¼ �2:27 – 0:455 CNT – 0:0784 PS þ 0:0552 NT – 0:0306 CNT � CNT þ 0:000338PS � PS – 0:000163 NT � NT þ 0:00356 CNT � NT þ 0:000213 PS � NT: ð3Þ 

In order to verify the adequacy of developed regression models, the variable of R[2] , R[2] Adj, and R[2] Pred were applied. The predictive ability of a regression model increases as the values of R[2] , R[2] Adj, and R[2] Pred approach 100%. In Tables 6 and 7, it is obvious that the variable of R[2] , R[2] Adj, and R[2] Pred for UTS and IE are near to 100%, indicating that the attained models are capable of estimating the UTS and IE. 

The impact of CNT content (CNT), PS, and NT on the IE and UTS of the PLA/CNT composite was shown in 

Figure 8. According to Figure 8, the greatest tensile strength of the PLA/CNT composite was achieved with 2 wt% CNT (Figure 8A), but with 4 wt% CNT, the greatest impact energy of the composite was achieved (Figure 8B). The variations in the impact energy and tensile strength of the PLA/CNT composite can be explained based on SEM images in Figure 5. The increase of the tensile strength of the composite with 2 wt% CNT is because of the fine dispersion of the CNTs and thus the strong interfacial interaction between the nanoparticles and PLA. Nevertheless, the aggregation of the nanoparticles in the composite with 4 wt% CNT decreased the tensile strength (Figure 5C). The reduction in the tensile strength of the printed PLA with 4 wt% CNT may also be due to its increased crystallinity. It should be noted that the increase in crystallinity increases the internal rigidity of PLA and sometime reduces the overall strength of the specimen. The impact energy of the composite was also improved at CNT content of 4 wt% because some of the impact energy is absorbed by the CNT nanoparticles. Similar results were observed in.[6,13,41,42] According to Figure 8, the impact energy and tensile strength of the composite decreased by about 12 and 22%, respectively, with an increment in the PS from 20 to 60 mm/s. An increase in the impact energy and tensile strength at PS of 20 mm/s is because of the creation of the good 

HARDANI ET AL. 

5055 

TABLE 6 ANOVA for tensile strength. 

|Source|DF|Adjusted SS|Adjusted MS|F|p|Contribution (%)|
|---|---|---|---|---|---|---|
|Model|9|936.039|104.004|77.68|0.000|-|
|Linear|3|205.118|68.373|51.07|0.000|-|
|CNT|1|69.915|69.915|52.22|0.001|7.19|
|PS|1|66.875|66.875|49.95|0.001|6.88|
|NT|1|68.328|68.328|51.03|0.001|7.03|
|Square|3|698.479|232.826|173.90|0.000|-|
|CNT�CNT|1|676.583|676.583|505.34|0.000|69.63|
|PS�PS|1|19.307|19.307|14.42|0.013|1.97|
|NT�NT|1|38.026|38.026|28.40|0.003|3.91|
|2-Way interaction|3|32.442|10.814|8.08|0.023|-|
|CNT�PS|1|20.976|20.976|15.67|0.011|2.15|
|CNT�NT|1|0.007|0.007|0.01|0.944|0.00|
|PS�NT|1|11.458|11.458|8.56|0.033|1.18|
|Error|5|6.694|1.339|||0.06|
|Lack-of-fit|3|6.504|2.168|22.81|0.042|-|
|Pure error|2|0.190|0.095|||-|
|Total|14|942.734||||-|
|R2 =99.29%||R2Adj=98.01%||R2Pred=88.92%|||



TABLE 7 ANOVA for impact energy. 

|Source|DF|Adjusted SS|Adjusted MS|F|p|Contribution (%)|
|---|---|---|---|---|---|---|
|Model|9|1.06476|0.118307|63.78|0.000|-|
|Linear|3|0.80132|0.267108|143.99|0.000|-|
|CNT|1|0.33620|0.336200|181.24|0.000|31.66|
|PS|1|0.35701|0.357013|192.46|0.000|33.64|
|NT|1|0.10811|0.108113|58.28|0.001|10.17|
|Square|3|0.14909|0.049697|26.79|0.002|-|
|CNT�CNT|1|0.05541|0.055408|29.87|0.003|5.16|
|PS�PS|1|0.06729|0.067292|36.28|0.002|6.33|
|NT�NT|1|0.01560|0.015600|8.41|0.034|1.47|
|2-Way interaction|3|0.11435|0.038117|20.55|0.003|-|
|CNT�PS|1|0.00423|0.004225|2.28|0.192|0.39|
|CNT�NT|1|0.08122|0.081225|43.79|0.001|7.65|
|PS�NT|1|0.02890|0.028900|15.58|0.011|2.72|
|Error|5|0.00927|0.001855|||0.82|
|Lack-of-fit|3|0.00667|0.002225|1.71|0.389|-|
|Pure error|2|0.00260|0.001300|||-|
|Total|14|1.07404||||-|
|R2 =99.39%||R2Adj=98.30%||R2Pred=91.50%|||



adhesion bonding between filaments and consequently the reduction of cavities and cracks in the microstructure (Figure 6A). However, an increment of PS up to 60 mm/s 

decreased the bonding strength between filaments, which resulted in the generation of defects in the microstructure and thus the decrease of the impact energy and tensile 

HARDANI ET AL. 

5056 

**==> picture [501 x 160] intentionally omitted <==**

FIGURE 8 Effect of FFF parameters on (A) tensile strength and (B) impact energy. 

strength of PLA/CNT composite (Figure 6C). The decrease of the strength and impact energy of PLA at higher PS was also reported in.[43][–][47] 

Figure 8 depicts that an increase in the NT from 180 to 200[�] C was followed by a rise in the tensile strength and impact energy by 12 and 9%, respectively. However, a rise in NT to 220[�] C improved the impact energy by 3% and declined the tensile strength by 2%. According to SEM image in Figure 7B, the higher adhesion of the printed filaments owing to the lower viscosity of the PLA is responsible for the higher impact energy and tensile strength at NT of 200[�] C. The strong adhesion between the filament layers at 200[�] C decreases the existence of cavities and cracks (Figure 7B), giving rise to a rise in the impact energy and tensile strength of the composite. Additionally, based on SEM image in Figure 7A, the formation of defects in the microstructure due to the poor adhesion of the filament layers deteriorated the impact energy and tensile strength at the NT of 180[�] C. It worth mentioning that when the temperature between the deposited filaments is not sufficient, a weak interlayer bond is formed due to the generation of a low molecular chain fusion between the filaments.[48] The attained results are in good agreement with those observed in.[25,30,31,43,49][–][52] 

The concurrent effect of the parameters on the impact energy and tensile strength was studied by considering the significant interaction in the ANOVA tables. Figure 9A,B exhibit the interaction effect of CNT � PS and PS � NT on the tensile strength, and Figure 9C,D exhibit the interaction effect of CNT � NT and PS � NT on the impact energy. Figure 9A indicates that when CNT content is 0 wt%, a rise of PS has a trivial effect on the tensile strength, but when CNT content is 4 wt%, a rise of PS decreased the tensile strength because increasing the PS reduces the generated heat in the polymer and some of the heat is absorbed by CNT nanoparticles. However, the 

greatest tensile strength of the composite was achieved at PS of 60 mm/s and CNT content of 2 wt%. It should be explained that when the CNT content is 2 wt%, increasing the PS up to 60 mm/s causes the CNTs and polymer chains to be oriented in the direction of the applied force, which leads to an increase of the tensile strength of the composite.[53,54] In addition, a rise of the CNT content up to 4 wt% leads to the aggregation of nanoparticles. Therefore, with increasing the PS, the CNTs and polymer chains are not oriented in the direction of applied tensile stress, which leads to a decline in the tensile strength. 

Figure 9B indicates that when PS is 20 mm/s, the tensile strength increased a little by enhancing the NT, but when PS is 60 mm/s, the tensile strength increased sharply by enhancing the NT. However, the maximum tensile strength of the composite was observed at NT of 220[�] C and PS of 20 mm/s. According to Figure 9C, for CNT content of 0 wt%, the impact energy of the composite slightly reduced with an increase of NT, while for CNT content of 4 wt%, the impact energy of the composite considerably increased with an increase of NT. The enhancement of impact energy of the composite at NT of 220[�] C and CNT content of 4 wt% is owing to the prevention of thermal deterioration of the composite by CNTs. In fact, a part of generated heat is absorbed by CNTs. Figure 9D represents that as the PS is 20 mm/s, the increase of NT slightly enhanced the impact energy, but as the PS is 60 mm/s, the increase of NT considerably enhanced the impact energy. 

The UF technique was employed to concurrently enhance the impact energy and tensile strength of the PLA/CNT composite. According to this technique, the output responses of the impact energy and tensile strength are first transformed to the individual utility function (d). After that, the geometric average of the individual utility (d) is attained to estimate the combined utility function (D). Eventually, the desirable condition of 

HARDANI ET AL. 

5057 

**==> picture [501 x 359] intentionally omitted <==**

FIGURE 9 Interaction of parameters for (A,B) tensile strength and (C,D) impact energy. 

the parameters is identified by maximizing the combined utility functions. The utility function can have a value between zero and one, the optimum value of which is one. Different objectives is usually considered for each response to compute the individual utility function: 1—a — certain response value is better; 2 lower response is better; 3—higher response is better.[18,55,56] The objective of this paper is to obtain greater values of the impact energy and tensile strength of the PLA/CNT composite, thus for estimating the individual utility function, the first objective is selected. For the objective of “higher is better,” the following equation can be calculated to achieve the individual utility function (di): 

**==> picture [185 x 49] intentionally omitted <==**

where O denotes the objective response value (x), L denotes the lower response value (x), and t-value represents the weighting factor of a response (x). By 

computing the geometric average of individual utility functions (di) for tensile strength and impact energy, the combined utility function (D) can also be achieved as follows: 

**==> picture [230 x 21] intentionally omitted <==**

In Equation (5), d1(Tensile strength) depicts the individual utility function for tensile strength, d2(Impact energy) depicts the individual utility function for impact energy, and D represents the combined utility function.[18,55,56] In the present work, Minitab[®] 17 software is applied to determine the individual and combined utility functions. Figure 10 displays the optimization plot obtained by utility function technique. The optimization plot in Figure 10 indicates that a high combined utility has been obtained (D = 0.99), representing that the estimated impact energy and tensile strength are close to their maximum value (objective value). Furthermore, Figure 10 indicates that the impact energy and tensile strength of the composite can be improved by CNT 

HARDANI ET AL. 

5058 

**==> picture [370 x 314] intentionally omitted <==**

FIGURE 10 Optimization results by utility function. 

|Responses<br>Prediction values<br>Verification values<br>Error<br>Tensile strength (MPa)<br>62.5<br>63.9<br>2.2%<br>Impact energy (J)<br>2.14<br>2.03<br>5.1%|TABLE 8<br>Comparison of prediction<br>results and verification tests at the<br>optimum conditions.|
|---|---|



content of 2.8 wt%, PS of 20 mm/s, and NT of 209[�] C. It was also predicted that the tensile strength and impact energy of the PLA/CNT composite increased up to 2.14 J and 62.5 MPa at the optimum conditions. 

Several verification experiments were done at the optimum conditions to validate the accuracy of the optimization results. The results of the prediction and verification experiments were listed in Table 8. It is concluded from the results in Table 8 that the verification experiments are in good agreement with the prediction results. It should be stated that the PLA/CNT composite, fabricated in optimal conditions, can be used in automobile body such as body panels, covers, and bumper due to its desirable tensile strength and impact resistance. 

## 4 | CONCLUSION 

In the present work, the impact energy and tensile strength of the PLA/CNT composite optimized concurrently by using the RSM and UF techniques. The thermal analysis of 

the PLA/CNT composite using the DSC test demonstrated that the melting and crystallization temperatures of the composite improved at the CNT content of 2 and 4 wt%. Moreover, the thermogravimetric analysis indicated that by adding the CNTs into the PLA, the thermal stability of the PLA/CNT composite increased. The greatest tensile strength of the composite was attained by adding 2 wt% CNTs owing to well dispersion of the nanoparticles inside the PLA, while the impact energy of the composite was maximized by addition of 4 wt% CNTs owing to the absorption of some energy by nanoparticles. The tensile strength and impact energy of the composite decreased by improving the speed of printing to 60 mm/s due to the creation of a weak adhesion bonding between the printed layers and consequently the generation of cavities and cracks in the microstructure. A rise in the temperature of nozzle to 220[�] C increased the impact energy and tensile strength of the composite because of the formation of strong adhesion between the printed filaments. The results of optimization demonstrated that the tensile strength and impact energy of the composite enhanced simultaneously 

HARDANI ET AL. 

5059 

at a nozzle temperature of 209[�] C, print speed of 20 mm/s, and CNT content of 2.8 wt%. Under this optimal condition, the tensile strength and impact energy were estimated to be 62.5 MPa and 2.14 J, respectively. 

CONFLICT OF INTEREST STATEMENT There is no conflict of interest to declare. 

## DATA AVAILABILITY STATEMENT 

The authors confirm that the data supporting the findings of this study are available within the article. 

## ORCID 

Mahmoud Afshari https://orcid.org/0000-0002-85564264 

## REFERENCES 

1. Petousis M, Vidakis N, Mountakis N, et al. Multifunctional material extrusion 3D-printed antibacterial Polylactic acid (PLA) with binary inclusions: the effect of cuprous oxide and cellulose nanofibers. Fibers. 2022;10:52. 

2. Divakara Shetty S, Shetty N. Investigation of mechanical properties and applications of polylactic acids—a review. Mater Res Express. 2019;6:112002. 

3. Singhvi MS, Zinjarde SS, Gokhale DV. Polylactic acid: synthesis and biomedical applications. J Appl Microbiol. 2019;127: 1612-1626. 

4. Hamad K, Kaseem M, Yang HW, Deri F, Ko YG. Properties and medical applications of polylactic acid: a review. Express Polym Lett. 2015;9:435-455. 

5. Casalini T, Rossi F, Castrovinci A, Perale G. A perspective on Polylactic acid-based polymers use for nanoparticles synthesis and applications. Front Bioeng Biotechnol. 2019;7:259. 

6. Zhou Y, Lei L, Yang B, Li J, Ren J. Preparation and characterization of polylactic acid (PLA) carbon nanotube nanocomposites. Polym Test. 2018;68:34-38. 

7. Rosli NA, Ahmad I, Anuar FH, Abdullah I. Mechanical and thermal properties of natural rubber-modified poly(lactic acid) compatibilized with telechelic liquid natural rubber. Polym Test. 2016;54:196-202. 

8. Vidakis N, Petousis M, Kourinou M, Velidakis E, Mountakis N. Peder Erik Fischer-Griffiths, Sotirios Grammatikos, Lazaros Tzounis, additive manufacturing of multifunctional polylactic acid (PLA)—multiwalled carbon nanotubes (MWCNTs) nanocomposites. Nanocomposites. 2021;7:184-199. 

9. Pinto VC, Ramos T, Sofia Alves J, Xavier PT, Moreira PMGP, Guedes RM. Comparative failure analysis of PLA, PLA/GNP and PLA/CNT-COOH biodegradable nanocomposites thin films. Procedia Eng. 2015;114:635-642. 

10. Doagou-Rad S, Islam A, Jensen JS. Influence of processing conditions on the mechanical behavior of MWCNT reinforced thermoplastic nanocomposites. Procedia CIRP. 2017;66:131-136. 

11. Mohd Nurazzi N, Asyraf MRM, Khalina A, et al. Fabrication, functionalization, and application of carbon nanotube-reinforced polymer composite: an overview. Polymers. 2021;13:1047. 

12. Zhang D, Villarreal MG, Cabrera E, Benatar A, James Lee L, Castro JM. Performance study of ultrasonic assisted processing 

   - of CNT nanopaper/solventless epoxy composite. Compos Part B. 2019;159:327-335. 

13. Yang L, Lia S, Zhou X, et al. Effects of carbon nanotube on the thermal, mechanical, and electrical properties of PLA/CNT printed parts in the FDM process. Synth Met. 2019;253:122-130. 

14. Pinto VC, Ramos T, Alves ASF, et al. Dispersion and failure analysis of PLA, PLA/GNP and PLA/CNT-COOH biodegradable nanocomposites by SEM and DIC inspection. Eng Fail Anal. 2017;71:63-71. 

15. Ivanov E, Kotsilkova R, Xia H, Chen Y, Donato RK, Donato K. Anna Paula Godoy, Rosa Di Maio, Clara Silvestre, Sossio Cimmino, Verislav Angelov, PLA/graphene/MWCNT composites with improved electrical and thermal properties suitable for FDM 3D printing applications. Appl Sci. 2019;9:1209. 

16. Zhenzhen X, Dou T, Wang Y, et al. Three-dimensional-printed carbon nanotube/Polylactic acid composite for efficient electromagnetic interference shielding. Polymers. 2023;15:3080. 

17. Batakliev T, Georgiev V, Angelov V, et al. Synergistic effect of graphene Nanoplatelets and multiwall carbon nanotubes incorporated in PLA matrix: nanoindentation of composites with improved mechanical properties. J Mater Eng Perform. 2021;30: 3822-3830. 

18. Taher F, Afshari M, Houmani A. Mohammad Reza Samadi, Soroush Bakhshi, Hossein Afshari, simultaneous enhancement of the impact strength and tensile modulus of PP/EPDM/TiO2 nanocomposite fabricated by fused filament fabrication. Colloid Polym Sci. 2024;302:393-407. 

19. Goh GD, Yap YL, Tan HKJ, Sing SL, Goh GL, Yeong WY. Process–structure–properties in polymer additive manufacturing via material extrusion: a review. Crit Rev Solid State Mater Sci. 2020;45(2):113-133. 

20. Anis A, Ansari A, Kamil M. Effect of print speed and extrusion temperature on properties of 3D printed PLA using fused deposition modeling process. Mater Today: Proc. 2021;45:5462-5468. 

21. Farazin A, Mohammadimehr M. Effect of different parameters on the tensile properties of printed Polylactic acid samples by FDM: experimental design tested with MDs simulation. Int J Adv Manuf Technol. 2022;118:103-118. 

22. Chacon JM, Caminero MA, Garcia-Plaza E, Nunez PJ. Additive manufacturing of PLA structures using fused deposition modelling: effect of process parameters on mechanical properties and their optimal selection. Mater Des. 2017;124:143-157. 

23. Khosravani MR, Reinicke T. Effects of raster layup and printing speed on strength of 3D-printed structural components. In Proceedings of the 1st European-Structural-Integrity-Society (ESIS) Virtual European Conference on Fracture (ECF), Virtual, 29 June–1 July (2020) pp. 720–725. 2020. 

24. Kuznetsov VE, Solonin AN, Tavitov A, Urzhumtsev O, Vakulik A. Increasing strength of FFF three-dimensional printed parts by influencing on temperature-related parameters of the process. Rapid Prototyp J. 2020;26:107-121. 

25. Altan M, Eryildiz M, Gumus B, Kahraman Y. Effects of process parameters on the quality of PLA products fabricated by fused deposition modeling (FDM): surface roughness and tensile strength. Mater Test. 2018;60:471-477. 

26. Cardoso PHM, Teixeira BN, Calado VMD, et al. Mechanical and dimensional performance of poly(lactic acid) 3D-printed parts using thin plate spline interpolation. J Appl Polym Sci. 2020;137:49171. 

HARDANI ET AL. 

5060 

27. Benwood C, Anstey A, Andrzejewski J, Misra M, Mohanty AK. Improving the impact strength and heat resistance of 3D printed models: structure, property, and processing Correlationships during fused deposition modeling (FDM) of poly(lactic acid). ACS Omega. 2018;3:4400-4411. 

28. Hsueh MH, Lai CJ, Wang SH, et al. Effect of printing parameters on the thermal and mechanical properties of 3D-printed PLA and PETG using fused deposition modeling. Polymers. 2021;13:1758. 

29. Wach RA, Wolszczak P, Adamus-Wlodarczyk A. Enhancement of mechanical properties of FDM-PLA parts via thermal annealing. Macromol Mater Eng. 2018;303:9. 

30. Liparoti S, Sofia D, Romano A, Marra F, Pantani R. Fused filament deposition of PLA: the role of interlayer adhesion in the mechanical performances. Polymers. 2021;13:399. 

31. Vanaei HR, Shirinbayan M, Vanaei S, Fitoussi J, Khelladi S, Tcharkhtchi A. Multi-scale damage analysis and fatigue behavior of PLA manufactured by fused deposition modeling (FDM). Rapid Prototyp J. 2021;27:371-378. 

32. Deshwal S, Kumar A, Chhabra D. Exercising hybrid statistical tools GA-RSM, GA-ANN and GA-ANFIS to optimize FDM process parameters for tensile strength improvement. CIRP J Manuf Sci Technol. 2020;31:189-199. 

33. Mora A, Verma P, Kumar S. Electrical conductivity of CNT/polymer composites: 3D printing, measurements and modeling. Compos Part B Eng. 2020;183:107600. 

34. Pietrzak L, Raniszewski G, Szymanski L. Multiwalled carbon nanotubes Polylactide composites for electrical engineering— fabrication and electrical properties. Electronics. 2022;11:3180. 

35. Zhaohua X, Niu Y, Wang Z, et al. Enhanced nucleation rate of Polylactide in composites assisted by surface acid oxidized carbon nanotubes of different aspect ratios. ACS Appl Mater Interfaces. 2011;3:3744-3753. 

36. Banerjee R. Suprakas Sinha ray, an overview of the recent advances in polylactide-based sustainable nanocomposites. Polym Eng Sci. 2021;61:617-649. 

37. Qi X-d, Yang J-h, Zhang N, et al. Selective localization of carbon nanotubes and its effect on the structure and properties of polymer blends. Prog Polym Sci. 2021;123:101471. 

38. Sang Ho Park. Seung goo Lee, Seong Hun Kim, isothermal crystallization behavior and mechanical properties of polylactide/carbon nanotube nanocomposites. Compos A: Appl Sci Manuf. 2013;46: 11-18. 

39. Ginoux G, Vroman I, Alix S. Influence of fused filament fabrication parameters on tensile properties of polylactide/layered silicate nanocomposite using response surface methodology. J Appl Polym Sci. 2021;138:50174. 

40. Das A, Gilmer EL, Biria S, Bortner MJ. Importance of polymer rheology on material extrusion additive manufacturing: correlating process physics to print properties. ACS Appl Polym Mater. 2021;3:1218-1249. 

41. Urquijo J, Aranburu N, Dagréou S, Guerrica-Echevarría G, Eguiaz�abal JI. CNT-induced morphology and its effect on properties in PLA/PBAT-based nanocomposites. Eur Polym J. 2017;93:545-555. 

42. Wang Y, Mei Y, Wang Q, et al. Improved fracture toughness and ductility of PLA composites by incorporating a small amount of surface-modified helical carbon nanotubes. Compos Part B Eng. 2019;162:54-61. 

43. Luzanin O, Movrin D, Stathopoulos V, Pandis P, Radusin T, Guduric V. Impact of processing parameters on tensile strength, 

   - in-process crystallinity and mesostructure in FDM-fabricated PLA specimens. Rapid Prototyp J. 2019;25:1398-1410. 

44. Vanaei H, Shirinbayan M, Deligant M, et al. Influence of process parameters on thermal and mechanical properties of polylactic acid fabricated by fused filament fabrication. Polym Eng Sci. 2020;60:1822-1831. 

45. Hikmat M, Rostam S, Ahmed YM. Investigation of tensile property-based Taguchi method of PLA parts fabricated by FDM 3D printing technology. Results Eng. 2021;11:100264. 

46. Naveed N. Investigating the material properties and microstructural changes of fused filament fabricated PLA and toughPLA parts. Polymers. 2021;13:1487. 

47. Zakaria H, Khan SF, Fee MFC, Ibrahim M. Printing temperature, printing speed and raster angle variation effect in fused filament fabrication. In Proceedings of the 6th International Conference on Applications and Design in Mechanical Engineering (ICADME), – 

Penang, Malaysia, 26 27 August 2019; Volume 670. 2019. 

48. Bakhtiari H, Nikzad M, Tolouei-Rad M. Influence of threedimensional printing parameters on compressive properties and surface smoothness of Polylactic acid specimens. Polymers. 2023;15:3827. 

49. Bayraktar O, Uzun G, Cakiroglu R, Guldas A. Experimental study on the 3D-printed plastic parts and predicting the mechanical properties using artificial neural networks. Polym Adv Technol. 2017;28:1044-1051. 

50. Alafaghani A, Qattawi A, Alrawi B, Guzman A. Experimental optimization of fused deposition modelling processing parameters: a design-for-manufacturing approach. Procedia Manuf. 2017;10:791-803. 

51. Alafaghani A, Qattawi A. Investigating the effect of fused deposition modeling processing parameters using Taguchi design of experiment method. J Manuf Process. 2018;36:164-174. 

52. Attolico MA, Casavola C, Cazzato A, Moramarco V, Renna G. Effect of extrusion temperature on fused filament fabrication parts orthotropic behaviour. Rapid Prototyp J. 2020;26:639-647. 

53. Three-dimensional printing of poly(lactic acid) bio-based composites with sugarcane bagasse fiber: effect of printing orientation on tensile performance. Polym Adv Technol. 2019;30:910-922. 

54. Fundamentals of crystalline evolution and properties of carbon nanotube-reinforced polyether ether ketone nanocomposites in fused filament fabrication. ACS Appl Mater Interfaces. 2023;15: 22506-22523. 

55. Taher F, Hardani H, Bakhshi S, Samadi MR, Ayaz M. Enhancing the tensile properties of PA6/CNT nanocomposite in selective laser sintering process. Polym Compos. 2023;44:1290-1304. 

56. Samadi MR, Ayaz M, Afshari M, Afkar A. An investigation on the friction stir welding of PP/TiO2 nanocomposites for improving the tensile strength and hardness of the weld joint. Colloid Polym Sci. 2023;301:465-480. 

How to cite this article: Hardani H, Afshari M, Allahyari F, Samadi MR, Afshari H, Medina EMM. Optimization of FFF process parameters to improve the tensile strength and impact energy of polylactic acid/carbon nanotube composite. Polym Eng Sci. 2024;64(10):5047‐5060. doi:10.1002/pen. 26900 

