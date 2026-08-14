# AcmeOps Integrations

AcmeOps includes native integrations for Slack, Microsoft Teams, Zendesk,
Salesforce, and Snowflake. The Slack integration can post incident summaries to
one or more channels and can also receive slash commands.

The Snowflake sync runs every six hours on the Growth plan. Enterprise customers
can request hourly Snowflake syncs.

Webhook retries use exponential backoff for up to 24 hours. Failed webhook
events remain visible in the developer console for seven days.

Metadata: product=acmeops, doc_type=integration, plan=growth
