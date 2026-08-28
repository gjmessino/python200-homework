----- Scenario A -----
Total Cost: $1.66

----- Scenario B -----
p3.2xlarge Cost: $2,233.80 
db.m5.large Cost: $322.30
S3 Standard Cost: $23.55

Total Cost: $2579.65

----- Comments -----
I was surprised by how cheap the total price was for Scenario A. I expected 160 of data usage to cost more. I expected it to be closer to the price of S3 Standard. As for the prices in Scenario B they had a much larger range than expacted, especially given that S3 required the most space, but was the cheapest. The EC system in this scenario cost 20x as much with only a fraction of the storage.

Notably these prices are strick to Amazon, and even within there, there is some wiggle room. There were at least 9 options for RDS depending on the exact needs of the user. I used Amazon RDS Custom for Oracle given that it was the top choice, but selecting other options would change the price, and potentially limit (or expand) capabilities. 

This shows that GPU servers in the P family are the most expensive, but are also made for building and training large data sets. Whereas, other servers maintain small sets of data and don't offer equivilant training systems, aiming instead to focus on storage.