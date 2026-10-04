----- Cloud Concepts Question 1 ----- 
Cloud computing means using someone elses hardware for storage as opposed to your own. 
This is most helpful with large scale products that have too much data for one of even 
a few computers. With cloud computing businesses can rent space from other companies at 
a rate that's cheaper than building and hosting one's own hardware.

----- Cloud Concepts Question 2 -----
Vertical scaling is making one system bigger to handle large information loads. Examples 
including adding more RAM, GPU, or CPU to a device to gain more storage. 
Horizontal scaling includes adding new machines and splitting work between them to handle 
large data loads. An example of horizontal scaling is creating a data center with hundreds 
of servers to handle large data.

Scenario 1 - This could be either vertical or horizontal because it's unclear what the 
intended method of scaling it is; either method could work depending on the desires of the company. 

Scenario 2 -- This is vertical scaling given that it's focused on giving one machine more memory.

Scenario 3 -- This is horizontal scaling because the goal is to split data between multiple machines

----- Cloud Concepts Question 3 -----
Gmail (SaaS) -- GMail requires almost nothing from users, and is built run and maintained by Google.

Azure Virtual Machines (IaaS) -- Azure Virtual Machines allows users to do everything themselves, including operating systems, building environments and handle security updates.

AWS S3 (Simple Storage Service) (IaaS) -- AWS S3 let's user's do everything except for hardware. However, other iterations of AWS fit into other categories.

GitHub Codespaces (PaaS) -- Codespaces manages hardware while letting users generate code and other functionalities on their service

Snowflake (SaaS) -- Snowflake is completely managed by itself without requiring any infrastructure from the user

Supabase (BaaS) -- Supabase gives users building blocks for coding while handling the backend themselves

Software as a Service (SaaS) requires the least from users, because the third-party company handles front/back end as well as hardware. Dropbox allows file sharing for users, but the users are not required to build any soft/hardware to manage or maintain the product.

Platform as a Service (PaaS) sits between SaaS and IaaS because it gives users the ability to create and manage software, but hardware and infrastructure are handled by the company itself. Google App Engine allows users to create their own apps, while handling the hardware themselves.

Infrastructure as a Service (IaaS) gives the most freedom to users, allowing them to handle every level of the build. Azure Virtual Machines allows users to fully manage their own services.

----- Cloud Concepts Question 4 -----
Managed Data Providers run on top of cloud providers. While the latter gives 
accuss for to a host of tools for IaaS, the first preassembles tools for the user 
and works more like and SaaS or PaaS. With Managed Data Providers, they take care 
of workloads and are already optimized for data analytics. Using a managed data 
provider means losing out on the ability to create custom infrastructure and data 
handling, but it makes it easier for users to design other tools on top of it. 
Choosing between a Managed Provider vs a cloud provider comes down to how much freedom 
users want to have over what they build.

----- Cloud Concepts Question 5 ----- 
Cloud computing is not recomended if you can all your data on a single machine. 
In this instance the cost of cloud computing isn't worth the pay off for what can be 
managed locally. The second situation mentioned is that there is a steep learning curve. 
It can take a lot longer to learn cloud software than another programming language, so 
for first time users its extremely time consuming.

----- Cloud Landscape Question 1 -----
Amazon Web Service (AWS) -- This is the oldest and largest service and is often used by large businesses.

Google Cloud Platform (GCP) -- This is best for data analysis and machine learning and is the prefered environment for large scale infrastructure.

Microsoft Azure -- This is largely used by governments and non-profits with users often coming to it through other Microsoft products because it does that best job at integrating it's other services into its cloud software.

----- Cloud Landscape Question 2 -----
Supabase grants easier access. In the past their were issues with students being able to log into azure and sometimes being locked out for days at a time.

The second reason is pedagogical fit. Supabase stores its data in tables, whereas azure uses opaque files.

Thirdly pipeline coherence. Supabase's table structure reinforces lessons from this course, specifically the concepts around data models.

In this instance choosing a cloud tool comes down to accessiblity and previous knowledge. Other factors in choosig a cloud service include price, specs specific to each project, or the amount of infrastructure users want to create.

----- Cloud Landscape Question 3 -----
1. Object Storage -- This is a simple fetch and retrieve and the user just needs enough space to put their data/files somewhere. AWS S3 offers object storage as a basic function.
2. ML Platform -- The user needs access to train their model, which Ml Platforms cover regardless of the fact that it will be shut down after 4 hours. Vertex AI offers a platform to train and deploy models.
3. Serverless Compute -- This allows the user to run their program without having to oversee scaling like they might with compute. GCP Cloud Functions offers this.
4. LLM API -- The user needs access to a large language model, and this API offers that without the added functions of an ML Platform. Azure Open AI can do this.

----- Cloud Landscape Question 4 -----
A project could be building an interface for an AI chatbot. LLM API would be necessary for 
accessing the AI and getting responses back, and integrating it with compute/serverless compute 
would allow for a more interactive UX with other functions.

When mixing cloud landscapes you can maximize productivity by getting the best platform for each 
service. If users like AWS bedrock for LLM but prefer Azure Virtual Machines for compute they can 
do this. The biggest limation in this instance is cost given different pricing ranges for different
companies. User's could also lose out of integration with a companies other services, like how 
Microsoft integrates their other services with cloud computing software.