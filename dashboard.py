import os,requests,streamlit as st,pandas as pd
st.set_page_config(page_title='RouteGuard AI',layout='wide')
st.title('RouteGuard AI · Explainable LLM Routing')
st.caption('Task complexity → model selection → generation → deterministic checks + independent LLM judge → quality-triggered escalation. Demo mode never claims verified quality.')
url=os.getenv('API_URL','http://localhost:8000').rstrip('/')
with st.form('run'):
    prompt=st.text_area('Your prompt',placeholder='Ask a question, summarize context, extract structured data or write code...',height=120)
    context=st.text_area('Optional grounding context (separate chunks with blank lines)',height=100)
    a,b,c=st.columns(3)
    quality=a.slider('Quality threshold',.5,1.,.9,.05)
    attempts=b.number_input('Maximum attempts',1,7,3)
    use_judge=c.checkbox('Independent LLM judge',True)
    expected=st.text_input('Optional reference answer')
    terms=st.text_input('Required terms (comma-separated)')
    expected_json=st.checkbox('Require valid JSON')
    submit=st.form_submit_button('Analyze, route and evaluate',type='primary')
if submit:
    if not prompt.strip():st.warning('Enter a prompt.')
    else:
        try:
            with st.spinner('Routing and evaluating...'):
                r=requests.post(url+'/route',json={'prompt':prompt,'context':context,'quality_threshold':quality,'max_attempts':attempts,'use_judge':use_judge,'expected_answer':expected or None,'expected_json':expected_json,'required_terms':[x.strip() for x in terms.split(',') if x.strip()]},timeout=300);r.raise_for_status();data=r.json()
            p=data['profile'];st.subheader('Routing decision');st.write('**Task:**',p['task'],' · **Complexity:**',p['complexity'],' · **Selected model:**',data['selected_model']);st.info(p['selection_reason'])
            x,y,z,w=st.columns(4);x.metric('Total tokens',data['total_tokens']);y.metric('Total cost',f"${data['total_cost_usd']:.6f}" if data['pricing_complete'] else 'Incomplete pricing');z.metric('Fallbacks',data['fallback_count']);w.metric('Quality status',data['status'])
            st.subheader('Candidate ranking');cdf=pd.DataFrame(data['candidates']); st.dataframe(cdf.sort_values(['eligible','cost_per_success_usd','estimated_generation_cost_usd'],ascending=[False,True,True],na_position='last'),use_container_width=True); st.caption('Benchmark-qualified models are ranked by cost per successful answer. With insufficient task-specific benchmark evidence, RouteGuard explicitly enters cold-start exploration mode and uses known estimated cost rather than invented quality scores.');st.subheader('Response');st.write(data['answer'] or 'No response returned')
            st.subheader('Execution trace')
            for i,at in enumerate(data['attempts'],1):
                with st.expander(f"Attempt {i}: {at['model']} · {'PASSED' if at.get('passed') else 'NOT VERIFIED / FAILED'}",expanded=True):
                    st.write('**Why:**',at['reason']);st.write('**Generation tokens:**',at.get('input_tokens',0),'input /',at.get('output_tokens',0),'output');
                    judge_data = at.get('judge', {})

                    st.write(
                        '**Evaluation cost:**',
                        judge_data.get('cost_usd', 0)
                    )

                    attempt_total = (
                        (at.get('generation_cost_usd') or 0)
                        + (judge_data.get('cost_usd') or 0)
                    )

                    st.write(
                        '**Attempt total cost:**',
                        attempt_total
                    )

                    judge_status = judge_data.get('status')

                    if judge_status == 'evaluated':
                        evaluation_strategy = 'LLM-as-a-Judge'
                    elif judge_status == 'not_required':
                        evaluation_strategy = 'Deterministic evaluation'
                    elif judge_status == 'demo_unverified':
                        evaluation_strategy = 'Demo / not evaluated'
                    elif judge_status == 'disabled':
                        evaluation_strategy = 'Judge disabled'
                    else:
                        evaluation_strategy = 'Not evaluated'

                    st.write(
                        '**Evaluation strategy:**',
                        evaluation_strategy
                    )
                                        
                    st.write('**Generation cost:**',at.get('generation_cost_usd',0));st.write('**Latency:**',at.get('latency_s',0));st.write('**Deterministic checks:**',at.get('checks',{}));st.write('**Judge:**',at.get('judge',{}));
                    if at.get('error'):st.error(at['error'])
            if not data['pricing_complete']:st.warning('Some model prices are missing. Total cost is incomplete; set prices in models.json.')
            if data['demo_mode']:st.warning('Demo mode: simulated answers and token counts; no real LLM judge or model calls.')
        except requests.RequestException as exc:st.error(f'API error: {exc}')
try:
    m=requests.get(url+'/metrics',timeout=5);m.raise_for_status();data=m.json();st.divider();st.subheader('Historical metrics');x,y,z=st.columns(3);x.metric('Runs',data['requests']);y.metric('Total recorded cost',f"${data['total_cost_usd']:.5f}");z.metric('Passed fraction',f"{data['verified_pass_rate']:.1%}");st.dataframe(pd.DataFrame(data['recent']),use_container_width=True)
except requests.RequestException:st.info('Start the API to display historical metrics.')

st.divider()
st.subheader('Offline benchmark performance')
st.caption('Run python -m routeguard.benchmark with real API credentials to populate these profiles. No simulated benchmark scores.')
try:
    resp=requests.get(url+'/benchmarks/profiles',timeout=5);resp.raise_for_status();profiles=resp.json()['profiles']
    if profiles:
        df=pd.DataFrame(profiles)
        st.dataframe(df,use_container_width=True)
        st.bar_chart(df.pivot_table(index='model',columns='task',values='wilson_lower_bound'))
    else:st.info('No empirical benchmarks yet. Run the benchmark CLI; heuristic scores are not validated quality predictions.')
except requests.RequestException:st.info('Benchmark profiles become available when the API is running.')
