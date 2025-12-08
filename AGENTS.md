This project is about analyzing git projcts. The gitview/cli.py orchestrates a series of actions: 

 - 1. retreaval of the git in question,
 - 2. read and chunk it up in different phases (based on preference)
 - 3. send each phase to a LLM for evaluation
 - 4. summarize the phases
 - 5. create a report
  

