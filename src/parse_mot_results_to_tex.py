# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Parse MOT results and generate a LaTeX table.
"""

MOTS = False
F_CONTENT = """
	MOTA	IDF1	MOTP	MT	ML	FP	FN	Recall	Precision	FAF	IDSW	Frag
    MOT17-01-DPM	41.6	44.2	77.1	5	8	496	3252	49.6	86.6	1.1	22	58
    MOT17-01-FRCNN	41.0	42.1	77.1	6	9	571	3207	50.3	85.0	1.3	25	61
    MOT17-01-SDP	41.8	44.3	76.8	7	8	612	3112	51.8	84.5	1.4	27	65
    MOT17-03-DPM	79.3	71.6	79.1	94	8	1142	20297	80.6	98.7	0.8	191	525
    MOT17-03-FRCNN	79.6	72.7	79.1	93	7	1234	19945	80.9	98.6	0.8	180	508
    MOT17-03-SDP	80.0	72.0	79.0	93	8	1223	19530	81.3	98.6	0.8	181	526
    MOT17-06-DPM	54.8	42.0	79.5	54	63	314	4839	58.9	95.7	0.3	175	244
    MOT17-06-FRCNN	55.6	42.9	79.3	57	59	363	4676	60.3	95.1	0.3	190	264
    MOT17-06-SDP	55.5	43.8	79.3	56	61	354	4712	60.0	95.2	0.3	181	262
    MOT17-07-DPM	44.8	42.0	76.6	11	16	1322	7851	53.5	87.2	2.6	147	275
    MOT17-07-FRCNN	45.5	41.5	76.6	13	15	1263	7785	53.9	87.8	2.5	156	289
    MOT17-07-SDP	45.2	42.4	76.6	13	15	1332	7775	54.0	87.3	2.7	147	279
    MOT17-08-DPM	26.5	32.2	83.0	11	37	378	15066	28.7	94.1	0.6	88	146
    MOT17-08-FRCNN	26.5	31.9	83.1	11	36	332	15113	28.5	94.8	0.5	89	141
    MOT17-08-SDP	26.6	32.3	83.1	11	36	350	15067	28.7	94.5	0.6	91	147
    MOT17-12-DPM	46.1	53.1	82.7	16	45	207	4434	48.8	95.3	0.2	30	50
    MOT17-12-FRCNN	46.1	52.6	82.6	15	45	197	4443	48.7	95.5	0.2	30	48
    MOT17-12-SDP	46.0	53.0	82.6	16	45	221	4426	48.9	95.0	0.2	30	52
    MOT17-14-DPM	31.6	36.6	74.8	13	78	636	11812	36.1	91.3	0.8	196	331
    MOT17-14-FRCNN	31.6	37.6	74.6	13	77	780	11653	37.0	89.8	1.0	202	350
    MOT17-14-SDP	31.7	37.1	74.7	13	76	749	11677	36.8	90.1	1.0	205	344
    OVERALL	61.5	59.6	78.9	621 	752	14076	200672	64.4	96.3	0.8	2583	4965
    """

# MOTS = True
# F_CONTENT = """
#     sMOTSA	MOTSA	MOTSP	IDF1	MT	ML	MTR	PTR	MLR	GT	TP	FP	FN	Rcll	Prcn	FM	FMR	IDSW	IDSWR
#     MOTS20-01	59.79	79.56	77.60	68.00	10	0	83.33	16.67	0.00	12	2742	255	364	88.28	91.49	37	41.91	16	18.1
#     MOTS20-06	63.91	78.72	82.85	65.14	115	22	60.53	27.89	11.58	190	8479	595	1335	86.40	93.44	218	252.32	158	182.9
#     MOTS20-07	43.17	58.52	76.59	53.60	15	17	25.86	44.83	29.31	58	8445	834	4433	65.58	91.01	177	269.91	75	114.4
#     MOTS20-12	62.04	74.64	84.93	76.83	41	9	60.29	26.47	13.24	68	5408	549	1063	83.57	90.78	76	90.94	29	34.7
#     OVERALL	54.86	69.92	80.62	63.58	181	48	55.18	30.18	14.63	328	25074	2233	7195	77.70	91.82	508	653.77	278	357.8
#     """

if __name__ == '__main__':
    # remove empty lines at start and beginning of F_CONTENT
    F_CONTENT = F_CONTENT.strip()
    F_CONTENT = F_CONTENT.splitlines()

    start_ixs = range(1, len(F_CONTENT) - 1, 3)
    if MOTS:
        start_ixs = range(1, len(F_CONTENT) - 1)

    metrics_res = {}

    for i in range(len(['DPM', 'FRCNN', 'SDP'])):
        for start in start_ixs:
            f_list = F_CONTENT[start + i].strip().split('\t')
            metrics_res[f_list[0]] = f_list[1:]

        if MOTS:
            break

    metrics_names = F_CONTENT[0].replace('\n', '').split()

    print(metrics_names)

    metrics_res['ALL'] = F_CONTENT[-1].strip().split('\t')[1:]

    for full_seq_name, data in metrics_res.items():
        seq_name = '-'.join(full_seq_name.split('-')[:2])
        detection_name = full_seq_name.split('-')[-1]

        if MOTS:
            print(f"{seq_name} & "
                f"{float(data[metrics_names.index('sMOTSA')]):.1f} & "
                f"{float(data[metrics_names.index('IDF1')]):.1f} & "
                f"{float(data[metrics_names.index('MOTSA')]):.1f} & "
                f"{data[metrics_names.index('FP')]} & "
                f"{data[metrics_names.index('FN')]} & "
                f"{data[metrics_names.index('IDSW')]} \\\\")
        else:
            print(f"{seq_name} & {detection_name} & "
                f"{float(data[metrics_names.index('MOTA')]):.1f} & "
                f"{float(data[metrics_names.index('IDF1')]):.1f} & "
                f"{data[metrics_names.index('MT')]} & "
                f"{data[metrics_names.index('ML')]} & "
                f"{data[metrics_names.index('FP')]} & "
                f"{data[metrics_names.index('FN')]} & "
                f"{data[metrics_names.index('IDSW')]} \\\\")
