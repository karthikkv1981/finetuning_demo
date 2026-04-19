import json

data = [    
("Why is this report still not done?", "Could you please share an update on the report status when you have a moment?"),
("This data is wrong. Fix it.", "I noticed some inconsistencies in the data—could you please review and update it?"),
("You missed the deadline again.", "I see the deadline was missed—let’s discuss any blockers and how we can stay on track moving forward."),
("This presentation is terrible.", "Thank you for your effort on the presentation—could we refine a few sections to improve clarity and impact?"),
("Stop making these mistakes.", "I’ve noticed a few recurring issues—let’s review them together to ensure smoother outcomes going forward."),
("Why didn’t you respond to my email?", "I wanted to follow up on my previous email—could you please provide an update when possible?"),
("This is not my job.", "I believe this task may fall under a different role—happy to collaborate on finding the right owner."),
("Your code is messy.", "Could we refactor parts of the code to improve readability and maintainability?"),
("You don’t understand the requirements.", "Let’s revisit the requirements together to ensure we’re aligned."),
("This meeting is a waste of time.", "Could we reassess the meeting agenda to ensure it’s productive for everyone?"),
("Fix this ASAP.", "Could you please prioritize this task and address it at your earliest convenience?"),
("You always do this wrong.", "I’ve noticed a pattern—let’s work together to find a consistent approach moving forward."),
("I already told you this.", "Just to reiterate, here are the details—please let me know if anything needs clarification."),
("This makes no sense.", "Could you please provide more context to help clarify this point?"),
("Don’t send me incomplete work.", "Could you please ensure the work is complete before sharing it for review?"),
("You’re slowing the team down.", "Let’s identify any challenges you’re facing so we can help maintain team momentum."),
("This isn’t good enough.", "Could we enhance this further to meet the expected standards?"),
("Why would you do it this way?", "Could you walk me through your approach so we can explore potential improvements?"),
("This needs a complete redo.", "I recommend revisiting this from the beginning to better align with the requirements."),
("I’m not happy with this.", "I think there’s an opportunity to improve this—happy to discuss suggestions."),
("You forgot an important detail.", "It seems a key detail may have been overlooked—could you please review?"),
("Stop bothering me with small issues.", "Could we consolidate minor issues into a single update for efficiency?"),
("This is confusing.", "Could you please clarify this section to make it easier to understand?"),
("You didn’t follow instructions.", "It seems there may have been a misalignment with the instructions—let’s review them together."),
("This is your fault.", "Let’s work together to identify what went wrong and how we can resolve it."),
("Don’t make the same mistake again.", "Let’s ensure we capture this learning to avoid similar issues in the future."),
("Why is this taking so long?", "Could you share an estimated timeline for completion?"),
("This isn’t what I asked for.", "I think there may be a mismatch with the requirements—could we align on expectations?"),
("You need to pay more attention.", "It would be helpful to focus on the details to ensure accuracy."),
("This looks unprofessional.", "Could we refine the formatting and presentation to make it more polished?"),
("You’re not prepared.", "Let’s ensure we’re fully prepared for future discussions by reviewing materials in advance."),
("I don’t care about this task.", "I’d like to better understand the importance of this task and how I can contribute effectively."),
("Fix your attitude.", "Let’s maintain a positive and collaborative approach in our interactions."),
("This is going nowhere.", "Let’s reassess our approach to ensure we’re making meaningful progress."),
("You’re wrong.", "I see it differently—could we discuss our perspectives to find the best solution?"),
("This email is unclear.", "Could you please provide more detail to make your message clearer?"),
("Why wasn’t I informed?", "I’d appreciate being kept in the loop on updates moving forward."),
("This is last-minute again.", "Could we plan ahead to avoid last-minute changes in the future?"),
("You didn’t do what I said.", "It seems there may have been a misunderstanding—could we review the expectations?"),
("This is too basic.", "Could we add more depth or detail to strengthen this work?"),
("I’m tired of fixing your work.", "Let’s collaborate on improving quality to reduce rework."),
("Don’t ignore my messages.", "I’d appreciate timely responses to ensure smooth communication."),
("You messed this up.", "There seem to be some issues—let’s review and resolve them together."),
("This isn’t useful.", "Could we adjust this to better meet the intended purpose?"),
("You’re overcomplicating things.", "Let’s simplify the approach to make it more efficient."),
("This doesn’t meet expectations.", "Could we revise this to better align with expectations?"),
("You’re not contributing enough.", "Let’s discuss how you can further contribute to the team’s goals."),
("This is poorly written.", "Could we refine the writing for clarity and professionalism?"),
("Why is this even included?", "Could you explain the relevance of this section?"),
("This needs to be better.", "There’s room for improvement—happy to provide feedback to enhance it.")
]

out_filename = "emails.jsonl"
with open(out_filename, "w") as f:
    for blunt, prof in data:
        record = {
            "instruction": f"Rewrite professionally: {blunt}",
            "output": prof
        }
        f.write(json.dumps(record) + "\n")

print(f"Generated {len(data)} examples in {out_filename}")
