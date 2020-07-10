# Summary
* Using natural language processing and deep learning techniques to write cover letters.

# Components
* Construct a general language model from an open data set.
* Create a data set of cover letters scraped from the internet.
* Use a generative adversarial network to write cover letters, start from initial language model and use cover letter data set to compare against.
* Create a data set of job titles and descriptions scraped from a job site.
* Build an associative model between job titles and required skills.

# Functionality
* Input = job title and json of personal skills.
* Use job model to determine which personal skills best fit the job.
* Use the cover letter model to write a letter which includes those skills.
* Add in decorations, address, signature, etc
* Compile to pdf
