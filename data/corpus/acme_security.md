# AcmeOps Security Controls

AcmeOps supports SAML 2.0 single sign-on on the Enterprise plan. Growth plan
customers can add SAML SSO as a paid security add-on.

Data is encrypted at rest using AES-256. Data in transit is protected with TLS
1.2 or newer. Audit logs record sign-in events, permission changes, webhook
updates, and export jobs.

Role-based access control has four default roles: Owner, Admin, Dispatcher, and
Viewer. Only Owners can delete a workspace.

Metadata: product=acmeops, doc_type=security, plan=enterprise
